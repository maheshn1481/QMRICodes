#!/usr/bin/env bash
# view_patient.sh -- open one patient's converted NIfTI set in ITK-SNAP.
#
# Usage:
#   ./view_patient.sh <AnonymizationID> [processed_dir]   # default dir: processed
#   ./view_patient.sh list [processed_dir]                # list available patients
#
# Loads T1W as the main (grayscale) image and the other modalities
# (FLAIR, T2W, T1map, T2map, PD) as overlays you can toggle in ITK-SNAP:
#   itksnap -g <main> -o <overlay1> <overlay2> ...
# Point [processed_dir] at processed_resampled/ to view the resampled set.
# Paths are AnonymizationID-based only (no PHI).
set -euo pipefail

id="${1:-}"
root="${2:-processed}"

if [ -z "$id" ] || [ "$id" = "-h" ] || [ "$id" = "--help" ]; then
    echo "usage: $0 <AnonymizationID> [processed_dir]"
    echo "       $0 list [processed_dir]"
    exit 1
fi

if [ "$id" = "list" ]; then
    [ -d "$root" ] || { echo "no such dir: $root" >&2; exit 1; }
    ls -1 "$root"
    exit 0
fi

command -v itksnap >/dev/null 2>&1 || { echo "itksnap not found on PATH" >&2; exit 1; }

dir="$root/$id"
[ -d "$dir" ] || { echo "no such patient dir: $dir" >&2; exit 1; }

main="$dir/T1W.nii.gz"
[ -f "$main" ] || { echo "missing main image: $main" >&2; exit 1; }

overlays=()
for m in FLAIR T2W T1map T2map PD PS; do
    [ -f "$dir/$m.nii.gz" ] && overlays+=("$dir/$m.nii.gz")
done

if [ "${#overlays[@]}" -gt 0 ]; then
    set -x
    exec itksnap -g "$main" -o "${overlays[@]}"
else
    set -x
    exec itksnap -g "$main"
fi
