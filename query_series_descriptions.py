#!/usr/bin/env python3
"""Build a database of the series available on PACS for each study in the cohort CSV.

The cohort CSV (schema in doc/fulldataset.md) names one series per patient --
``Series UID (Ax MAGiC)`` -- but not what else the study contains. This script
issues one SERIES-level C-FIND per ``Study UID`` with dcmtk's ``findscu`` and
writes every returned series (UID, number, description, modality, protocol,
instance count) to a CSV keyed by ``AnonymizationID``.

It mirrors the command-line interface of ../dicomtransfer/batch_cmove.py (same
--aet/--aec/--host/--port/--has-header/--study-col/--retries/--retry-wait/
--timeout/--log/--dry-run), so the query and the retrieval tools are driven the
same way. The equivalent command issued per study is:

    findscu -v -S -aet <aet> -aec <aec> <host> <port> \\
            -k 0008,0052=SERIES -k 0020,000D=<studyUID> \\
            -k 0020,000E= -k 0008,103E= -k 0020,0011= -k 0008,0060= \\
            -k 0018,1030= -k 0020,1209= -X

``-X`` makes findscu write each C-FIND response as a DICOM file (rsp*.dcm) in a
temporary directory, which is read back with pydicom; if the installed findscu
is too old for -X, the verbose text dump is parsed instead.

Usage:

    python3 query_series_descriptions.py dataset.csv --has-header \\
        --aet MYAET --aec REMOTEAEC --host 192.168.1.10 --port 104 \\
        --out series_index.csv --summary

Requires: dcmtk (system package, provides findscu) and pydicom.

PHI protection (see ../radpathsandbox/CLAUDE.md): the console prints only
AnonymizationID, counts and status -- Study/Series UIDs are hidden unless
--show-uids is passed (local debugging only; never paste or commit its output).
MRN is never read. The series database written by --out DOES contain Study and
Series UIDs by necessity and must stay out of git (it is gitignored); the
--log status file and the --summary table are UID-free.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter

import pandas as pd

# ----------------------------------------------------------------------------
# The C-FIND identifier: matching keys (with a value) and return keys (empty).
# Column order of the output CSV follows RETURN_KEYS.
# ----------------------------------------------------------------------------
RETURN_KEYS = [
    ("0020,000E", "SeriesInstanceUID"),
    ("0020,0011", "SeriesNumber"),
    ("0008,103E", "SeriesDescription"),
    ("0008,0060", "Modality"),
    ("0018,1030", "ProtocolName"),
    ("0020,1209", "NumberOfSeriesRelatedInstances"),
]
OUT_HEADER = ["AnonymizationID", "StudyInstanceUID"] + [n for _, n in RETURN_KEYS]
LOG_HEADER = ["AnonymizationID", "result", "n_series", "message"]

# Candidate CSV column names, used when --study-col/--id-col are left at default.
STUDY_COLS = ["Study UID", "StudyUID", "StudyInstanceUID", "study_uid", "Study Instance UID"]
ID_COLS = ["AnonymizationID", "Anonymization ID", "anon_id", "AnonID"]

# A DICOM UID is dotted digits; anything else in the study column (most often a
# header row read as data because --has-header was forgotten) is rejected.
UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)+$")

# One "(gggg,eeee) VR [value]" line of findscu's verbose dump (text fallback).
DUMP_RE = re.compile(
    r"^\s*[A-Z]:\s*\(([0-9a-fA-F]{4}),([0-9a-fA-F]{4})\)\s+\S\S\s+(?:\[(.*?)\]|\(no value available\))"
)


def clean(val) -> str:
    """Trim a DICOM value: odd-length UIDs come back NUL-padded, which str.strip()
    (whitespace only) leaves in place -- visible in findscu's text dump."""
    return str(val or "").replace("\x00", "").strip()


def redact(uid: str, show_uids: bool, label: str = "uid") -> str:
    """PHI-safe rendering of a UID: the literal only under --show-uids."""
    return uid if show_uids else f"<{label} hidden>"


# ----------------------------------------------------------------------------
# CSV input
# ----------------------------------------------------------------------------
def resolve_col(columns, requested: str, candidates, what: str) -> str:
    """Map a requested column name to the actual header (case-insensitive)."""
    lower = {str(c).strip().lower(): c for c in columns}
    for cand in [requested] + list(candidates):
        col = lower.get(str(cand).strip().lower())
        if col is not None:
            return col
    sys.exit(f"[error] {what} column {requested!r} not found; "
             f"available columns: {list(columns)}")


def read_studies(csv_path: str, has_header: bool, study_col: str, id_col: str):
    """Return a de-duplicated list of (anon_id, study_uid) plus skip counts."""
    if has_header:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        scol = resolve_col(df.columns, study_col, STUDY_COLS, "study")
        icol = resolve_col(df.columns, id_col, ID_COLS, "id")
    else:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, header=None)
        # headerless: the column flags are 0-based indices (batch_cmove.py
        # semantics); fall back to the dataset.csv layout if they are names.
        try:
            scol, icol = int(study_col), int(id_col)
        except ValueError:
            scol, icol = 2, 0  # AnonymizationID, MRN, Study UID, ...
        if max(scol, icol) >= df.shape[1]:
            sys.exit(f"[error] column index out of range for a {df.shape[1]}-column CSV")

    rows, seen = [], set()
    n_empty = n_bad = n_dup = 0
    for n, (_, row) in enumerate(df.iterrows(), start=2 if has_header else 1):
        study = str(row[scol]).strip()
        anon = str(row[icol]).strip()
        if not study:
            n_empty += 1
            continue
        if not UID_RE.match(study):
            print(f"[warn] row {n}: study value is not a DICOM UID, skipping"
                  + ("" if has_header else " (missing --has-header?)"))
            n_bad += 1
            continue
        key = (anon, study)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        rows.append((anon or f"row{n}", study))
    return rows, n_empty, n_bad, n_dup


# ----------------------------------------------------------------------------
# findscu
# ----------------------------------------------------------------------------
def findscu_command(args, study_uid: str) -> list:
    """Render the findscu command for one study's SERIES-level C-FIND."""
    cmd = [
        args.findscu, "-v",
        "-aet", args.aet,
        "-aec", args.aec,
        args.host, str(args.port),
        "-S",
        "-k", "0008,0052=SERIES",
        "-k", f"0020,000D={study_uid}",
    ]
    for tag, _name in RETURN_KEYS:
        cmd += ["-k", f"{tag}="]
    cmd += ["-X"]  # extract each response to rsp*.dcm in the working directory
    return cmd


def parse_rsp_files(tmpdir: str):
    """Read findscu's extracted rsp*.dcm responses with pydicom."""
    import pydicom  # imported here so --help/--dry-run work without it

    out = []
    for fn in sorted(os.listdir(tmpdir)):
        if not fn.lower().endswith(".dcm"):
            continue
        ds = pydicom.dcmread(os.path.join(tmpdir, fn), force=True)
        rec = {name: clean(getattr(ds, name, "")) for _, name in RETURN_KEYS}
        if rec["SeriesInstanceUID"]:
            out.append(rec)
    return out


def parse_dump(text: str):
    """Fallback: parse the verbose text dump when findscu lacks -X."""
    by_tag = {tag.lower().replace(",", ""): name for tag, name in RETURN_KEYS}
    out = []
    for block in text.split("Find Response:")[1:]:
        rec = {name: "" for _, name in RETURN_KEYS}
        for line in block.splitlines():
            m = DUMP_RE.match(line)
            if not m:
                continue
            name = by_tag.get((m.group(1) + m.group(2)).lower())
            if name:
                rec[name] = clean(m.group(3))
        if rec["SeriesInstanceUID"]:
            out.append(rec)
    return out


def find_one(args, study_uid: str, state: dict):
    """Run one C-FIND. Returns (ok, series_records, message)."""
    cmd = findscu_command(args, study_uid)
    with tempfile.TemporaryDirectory(prefix="findscu_") as tmpdir:
        try:
            proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True,
                                  text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            return False, [], f"timeout after {args.timeout}s"
        except FileNotFoundError:
            sys.exit(f"[error] findscu not found: {args.findscu!r} "
                     f"(install dcmtk, or pass --findscu /path/to/findscu)")

        try:
            series = parse_rsp_files(tmpdir)
        except ImportError:
            sys.exit("[error] pydicom is required to read findscu responses "
                     "(pip install -r requirements.txt)")

        used_fallback = False
        if not series:
            # No extracted responses: either the study genuinely has none, or
            # this findscu predates -X. Try the text dump before giving up.
            series = parse_dump((proc.stdout or "") + "\n" + (proc.stderr or ""))
            used_fallback = bool(series)

    if used_fallback and not state.get("warned_fallback"):
        state["warned_fallback"] = True
        print("[warn] findscu wrote no rsp*.dcm (-X unsupported?); "
              "falling back to parsing the verbose text dump")

    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        detail = (": " + err[-1]) if (err and args.show_uids) else " (--show-uids for detail)"
        return False, series, f"findscu exit {proc.returncode}{detail}"
    if not series:
        return False, [], "no series returned"
    return True, series, "ok"


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def load_done(out_path: str):
    """AnonymizationIDs already present in an existing --out file (for --resume)."""
    if not os.path.isfile(out_path):
        return set()
    with open(out_path, newline="") as f:
        return {row[0] for row in csv.reader(f) if row and row[0] != OUT_HEADER[0]}


def print_summary(out_path: str) -> None:
    """PHI-free histogram: series description -> number of patients having it.

    Read back from the database so the table covers the whole file, including
    patients carried over from an earlier (--resume) run.
    """
    per_patient = {}
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            desc = (row.get("SeriesDescription") or "").strip() or "(no description)"
            per_patient.setdefault(row["AnonymizationID"], set()).add(desc)
    counts = Counter(d for descs in per_patient.values() for d in descs)

    print(f"[summary] distinct series descriptions across {len(per_patient)} patient(s):")
    if not counts:
        print("[summary]   (none)")
        return
    width = max(len(d) for d in counts)
    for desc, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"[summary]   {desc:<{width}}  {n}")


def run(args) -> int:
    if not os.path.isfile(args.csv_file):
        sys.exit(f"[error] CSV not found: {args.csv_file!r}")

    rows, n_empty, n_bad, n_dup = read_studies(
        args.csv_file, args.has_header, args.study_col, args.id_col)
    if not rows:
        sys.exit("[error] no valid study UIDs found in the CSV")

    skipped = []
    if n_empty:
        skipped.append(f"{n_empty} empty")
    if n_bad:
        skipped.append(f"{n_bad} malformed")
    if n_dup:
        skipped.append(f"{n_dup} duplicate")
    print(f"[info] {len(rows)} studies to query"
          + (f" ({', '.join(skipped)} row(s) skipped)" if skipped else "")
          + ("" if args.show_uids else " (UIDs hidden; --show-uids to reveal)"))

    if args.dry_run:
        for i, (anon, study) in enumerate(rows, 1):
            shown = [c if c != f"0020,000D={study}"
                     else f"0020,000D={redact(study, args.show_uids, 'study uid')}"
                     for c in findscu_command(args, study)]
            print(f"[dry ] {anon}  " + " ".join(shown))
        return 0

    done = load_done(args.out) if args.resume else set()
    if done:
        print(f"[info] --resume: {len(done)} patient(s) already in {args.out}")

    n_ok = n_fail = n_series = n_skip = 0
    state = {}
    # --resume appends to whichever file already exists; a header is written
    # only when that file is being created.
    out_mode = "a" if (args.resume and os.path.isfile(args.out)) else "w"
    log_mode = "a" if (args.resume and os.path.isfile(args.log)) else "w"

    with open(args.out, out_mode, newline="") as outf, \
            open(args.log, log_mode, newline="") as logf:
        ow, lw = csv.writer(outf), csv.writer(logf)
        if out_mode == "w":
            ow.writerow(OUT_HEADER)
        if log_mode == "w":
            lw.writerow(LOG_HEADER)

        for i, (anon, study) in enumerate(rows, 1):
            if anon in done:
                print(f"[skip] {anon}  already queried (--resume)")
                n_skip += 1
                continue

            ok, series, msg = False, [], ""
            for attempt in range(1, args.retries + 2):
                ok, series, msg = find_one(args, study, state)
                if ok:
                    break
                if attempt <= args.retries:
                    print(f"[warn] {anon}  attempt {attempt} failed ({msg}); "
                          f"retrying in {args.retry_wait:.0f}s")
                    time.sleep(args.retry_wait)

            for rec in series:
                ow.writerow([anon, study] + [rec[n] for _, n in RETURN_KEYS])
            outf.flush()
            lw.writerow([anon, "OK" if ok else "FAIL", len(series), msg])
            logf.flush()

            tag = "find" if ok else "fail"
            print(f"[{tag}] {anon}  {len(series)} series  {msg}"
                  + (f"  study={study}" if args.show_uids else ""))
            n_series += len(series)
            n_ok += ok
            n_fail += (not ok)

    print(f"[done] studies={n_ok + n_fail} ok={n_ok} failed={n_fail} "
          + (f"skipped={n_skip} " if n_skip else "")
          + f"series={n_series}  out={args.out}  log={args.log}")
    if args.summary:
        print_summary(args.out)
    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_file", help="cohort CSV (doc/fulldataset.md schema)")
    ap.add_argument("--aet", required=True, help="Our calling AE title (findscu -aet)")
    ap.add_argument("--aec", required=True, help="Remote/source AE title of the PACS (findscu -aec)")
    ap.add_argument("--host", required=True, help="PACS IP/hostname")
    ap.add_argument("--port", type=int, required=True, help="PACS port")
    ap.add_argument("--has-header", action="store_true",
                    help="Treat the first CSV row as a header (dataset.csv has one)")
    ap.add_argument("--study-col", default="Study UID",
                    help="Column name (with header) or 0-based index for StudyInstanceUID")
    ap.add_argument("--id-col", default="AnonymizationID",
                    help="Column name (with header) or 0-based index for the PHI-free "
                         "patient key used in all console output")
    ap.add_argument("--out", default="series_index.csv",
                    help="Series database CSV (CONTAINS UIDs; keep out of git)")
    ap.add_argument("--log", default="series_find_log.csv",
                    help="Path to per-study result log (CSV, UID-free)")
    ap.add_argument("--findscu", default="findscu",
                    help="Path to the dcmtk findscu binary")
    ap.add_argument("--retries", type=int, default=2,
                    help="Retries per study on findscu failure")
    ap.add_argument("--retry-wait", type=float, default=5.0, help="Seconds between retries")
    ap.add_argument("--timeout", type=int, default=30, help="findscu timeout in seconds")
    ap.add_argument("--resume", action="store_true",
                    help="Skip patients already present in --out and append to it")
    ap.add_argument("--summary", action="store_true",
                    help="After the sweep, print the UID-free series-description "
                         "histogram (description -> number of patients)")
    ap.add_argument("--show-uids", action="store_true",
                    help="Reveal Study/Series UIDs in diagnostics (PHI); off by "
                         "default so console output stays PHI-safe")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the findscu command per study; do not associate")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
