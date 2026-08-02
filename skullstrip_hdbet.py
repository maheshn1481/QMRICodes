#!/usr/bin/env python3
"""Generate brain masks for the converted qMRI cohort using HD-BET.

The MATLAB qMRI pipeline uses an optional skull-strip **brain mask**
(``config.fileMask = "mask.nii.gz"``) to (a) keep only training patches with
>=20% brain coverage and (b) confine the loss / CCC metrics to the brain. Without
a mask it falls back to the nonzero-T1W support. This script produces a proper
brain mask per patient by running HD-BET (MIC-DKFZ/HD-BET) on the T1W image.

Run AFTER preprocess_dicom_to_nifti.py and BEFORE the MATLAB scripts. HD-BET is a
PyTorch model and runs on GPU by default when CUDA is available (seconds/case vs
minutes on CPU); the device is auto-selected (GPU 0 if present, else cpu):

    python3 skullstrip_hdbet.py --processed processed              # auto GPU/CPU
    python3 skullstrip_hdbet.py --processed processed --device 0   # force GPU index 0
    python3 skullstrip_hdbet.py --processed processed --fast       # ~8x faster (no TTA)

GPU needs a CUDA-enabled torch (installed with HD-BET on a CUDA machine).

For each <processed>/<AnonymizationID>/ that has T1W.nii.gz and no mask.nii.gz,
it runs HD-BET on T1W.nii.gz and writes the brain mask to
<processed>/<AnonymizationID>/mask.nii.gz. The mask shares the T1W grid, so it
matches the pipeline's reference grid (copied, not interpolated, by --resample)
and satisfies the per-patient size-equality checks.

Install HD-BET:  pip install HD-BET   (or from https://github.com/MIC-DKFZ/HD-BET)

PHI protection: only AnonymizationID and status are printed; no PHI-bearing paths.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


def find_hdbet_exe() -> str:
    """Locate the hd-bet console script. Checks next to the running interpreter
    first (its own venv bin -- pip installs the entry point there), then PATH. This
    matters when the script is launched via a full path to python without the venv
    activated, so /opt/<venv>/bin is not on PATH."""
    here = os.path.dirname(os.path.abspath(sys.executable))
    for name in ("hd-bet", "hd-bet.exe"):
        cand = os.path.join(here, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which("hd-bet") or ""


def default_device() -> str:
    """Pick GPU index '0' when CUDA is available, else 'cpu'. Uses '0' (not 'cuda')
    because both HD-BET v1 (integer index) and v2 accept an integer device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def find_mask(search_dir: str) -> str:
    """Locate the brain-mask NIfTI HD-BET wrote (naming varies across versions).
    Prefer names containing 'mask'; fall back to '*bet*' (v2 writes <out>_bet as the
    mask). The brain-extracted image is <out>.nii.gz and won't match these."""
    for pat in ("*_bet_mask.nii.gz", "*_mask.nii.gz", "*mask*.nii.gz", "*_bet.nii.gz"):
        hits = sorted(glob.glob(os.path.join(search_dir, pat)))
        if hits:
            return hits[0]
    return ""


def run_hdbet(exe: str, t1w: str, out_dir: str, device: str, fast: bool,
              capture: bool = False) -> str:
    """Run HD-BET on t1w into out_dir; return the produced brain-mask path.

    Tries HD-BET v2 then v1 command forms (their CLIs differ) until one succeeds and
    a mask is found. `device` is a GPU index ('0') or 'cpu'; `fast` disables
    test-time augmentation (v2 --disable_tta / v1 -tta 0 -mode fast) for ~8x speed.
    `capture=False` (default) streams HD-BET's own progress to the console;
    `capture=True` hides it and only surfaces the stderr tail on failure.
    """
    out = os.path.join(out_dir, "brain.nii.gz")
    env = os.environ.copy()
    # HD-BET v2 uses torch.device(), which rejects a bare index like "1"; select a
    # specific GPU via CUDA_VISIBLE_DEVICES and address it as the (remapped) first
    # visible device. 'cpu'/'cuda[:N]' are passed through unchanged.
    if str(device).lower() == "cpu":
        dev_v2 = dev_v1 = "cpu"
    elif str(device).lower().startswith("cuda"):
        dev_v2, dev_v1 = device, "0"
    else:  # bare integer index
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        dev_v2, dev_v1 = "cuda", "0"
    v2_speed = ["--disable_tta"] if fast else []
    v1_speed = ["-tta", "0", "-mode", "fast"] if fast else []
    variants = [
        [exe, "-i", t1w, "-o", out, "-device", dev_v2, *v2_speed, "--save_bet_mask"],  # v2
        [exe, "-i", t1w, "-o", out, "-device", dev_v1, *v1_speed, "-s", "1"],          # v1
        [exe, "-i", t1w, "-o", out, "-device", dev_v2],                                # plain
    ]
    last = None
    for cmd in variants:
        try:
            if capture:
                subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            else:
                subprocess.run(cmd, check=True, env=env)  # stream to console
        except subprocess.CalledProcessError as exc:
            last = exc
            continue
        mask = find_mask(out_dir)
        if mask:
            return mask
    if last is not None:
        tail = " | ".join((getattr(last, "stderr", "") or getattr(last, "stdout", "")
                           or "").strip().splitlines()[-2:])
        raise RuntimeError(tail or "hd-bet exited non-zero (see output above)")
    raise FileNotFoundError("HD-BET produced no mask")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="processed",
                    help="root with <AnonymizationID>/ subdirs (default: processed)")
    ap.add_argument("--main", default="T1W",
                    help="basename of the image to skull-strip (default: T1W)")
    ap.add_argument("--device", default=None,
                    help="HD-BET device: a GPU index (e.g. 0) or cpu. "
                         "Default: auto (GPU 0 if CUDA is available, else cpu)")
    ap.add_argument("--fast", action="store_true",
                    help="disable test-time augmentation for ~8x speedup "
                         "(slightly less accurate)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run even if mask.nii.gz already exists")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="only these AnonymizationIDs (default: all subdirs)")
    ap.add_argument("--quiet", action="store_true",
                    help="capture HD-BET's own output instead of streaming it "
                         "(default: stream so you see live progress)")
    args = ap.parse_args()

    device = args.device or default_device()
    print(f"[info] HD-BET device={device}"
          + (" (GPU)" if device not in ("cpu",) else " (CPU -- slow; use --device 0 "
             "on a CUDA box, add --fast to speed up)"))

    if not os.path.isdir(args.processed):
        sys.exit(f"[error] no such dir: {args.processed}")
    exe = find_hdbet_exe()
    if not exe:
        sys.exit("[error] 'hd-bet' not found next to this interpreter "
                 f"({os.path.dirname(sys.executable)}) or on PATH. Install into the "
                 "same env: pip install HD-BET (https://github.com/MIC-DKFZ/HD-BET)")
    print(f"[info] hd-bet: {exe}")

    ids = args.ids or sorted(d for d in os.listdir(args.processed)
                             if os.path.isdir(os.path.join(args.processed, d)))
    total = len(ids)
    print(f"[info] {total} patient(s) to process"
          + ("" if args.fast else "  (add --fast for ~8x speedup)"))
    if not args.quiet:
        print("[info] streaming HD-BET output below; --quiet to suppress")

    n_ok = n_skip = n_fail = 0
    for k, anon in enumerate(ids, 1):
        remaining = total - k
        pdir = os.path.join(args.processed, anon)
        t1w = os.path.join(pdir, args.main + ".nii.gz")
        mask = os.path.join(pdir, "mask.nii.gz")

        if not os.path.isfile(t1w):
            print(f"[{k}/{total}] {anon}: SKIP (no {args.main}.nii.gz)", flush=True)
            n_skip += 1
            continue
        if os.path.isfile(mask) and not args.overwrite:
            print(f"[{k}/{total}] {anon}: KEEP (mask exists)", flush=True)
            n_skip += 1
            continue

        print(f"[{k}/{total}] {anon}: skull-stripping on device {device}"
              f" ... ({remaining} left after this)", flush=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                produced = run_hdbet(exe, t1w, tmp, device, args.fast,
                                     capture=args.quiet)
                if not produced or not os.path.isfile(produced):
                    raise FileNotFoundError("HD-BET produced no mask")
                shutil.move(produced, mask)
            print(f"[{k}/{total}] {anon}: OK -> mask.nii.gz  "
                  f"(done {n_ok + 1}, {remaining} left)", flush=True)
            n_ok += 1
        except Exception as exc:  # never abort the whole cohort
            reason = str(exc).strip().replace("\n", " ")
            print(f"[{k}/{total}] {anon}: FAIL -- {reason[:300] or type(exc).__name__}",
                  flush=True)
            n_fail += 1

    print(f"[done] masked={n_ok} skipped/kept={n_skip} failed={n_fail} of {total}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
