#!/usr/bin/env python3
"""Classify the PACS series inventory into imaging series types and count them per study.

Consumes the series database written by query_series_descriptions.py
(series_index.csv) and labels each series by matching its SeriesDescription
against an ordered list of regular expressions, then reports how many series of
each type each study contains.

The rules are a direct port of the R ``seriesTypes`` list. ORDER IS LOAD-BEARING:
each series takes the FIRST rule it matches, which is what separates AxT1C from
the more general T1C and T1, and CubeFLAIR from the more general FLAIR. The
non-anatomical junk rule (nonAnat) is first so scouts, calibrations, functional
and localizer series are pulled out before any anatomical rule sees them.

Matching is case-insensitive by default (the R patterns mix conventions --
"Sag(?!.*REFORMAT).*CUBE.*FL", "probe", "^Exponen" -- so they only behave as one
rule set when case is ignored); pass --case-sensitive for literal R
``grepl(perl=TRUE)`` semantics.

Usage:

    python3 classify_series_types.py series_index.csv
    python3 classify_series_types.py series_index.csv --show-unmatched 40

Outputs:
    series_types.csv        every series with its assigned type (CONTAINS UIDs)
    series_type_counts.csv  per-study counts, one column per type (UID-free)

PHI: the console and the counts file carry only AnonymizationID, series
descriptions and counts -- no UIDs, no MRN. The per-series file necessarily
repeats the UIDs from series_index.csv and is gitignored.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd

# ----------------------------------------------------------------------------
# Series type rules, ported verbatim from the R seriesTypes list.
# ORDER MATTERS: first match wins, so specific rules must precede general ones
# (AxT1C before T1C before T1; CubeFLAIR before FLAIR; WandT2 before AxT2).
# R's "\\+" (an escaped backslash in an R string) is a plain "\+" here.
# Commented-out alternates from the R source are kept for provenance.
# ----------------------------------------------------------------------------
SERIES_TYPES = [
    ("nonAnat", r"(BRAIN SUITE|^DynaSuite|FUNCTIONAL|FUNCT|HAND$|FINGER|Speech|epiRT|fMRI"
                r"|HANDS|NECK|Multiplanar|ASSET|CAL|SCOUT|SCREEN|BURNED|LOC|CHEST|VEINS"
                r"|SELLA|REGISTRATION|COLOR|Anatomy|probe|MRS |^(CAT|FAS|SENT|MOTOR))"),

    ("T13D",   r"(3D.*WAND|WAND.*3D|(?<!R1[-_])SPGR|T1 Wand|\+C Ax 3D T1|T1 3D|3D AX T1 Stealth)"),
    # R alternate: "^(?!.*(SAG|COR)).*(3D.*wAND|WAND.*3D|SPGR|T1 Wand)"
    ("AxT1C",  r"(AX.*T1.*(POST|\+C)|POST-AXIAL)"),
    ("T1C",    r"(\+C.*T1|T1.*C|POST|SPGR)"),

    ("T1",     r"[^D]T1"),
    ("T2star", r"(T2\*|T2 \*|T2 ?STAR)"),
    # R alternate: SWAN = "SWAN"

    # "SAG.*FL.*CUBE" added 2020-03-05, affects PREOP only.
    ("CubeFLAIR", r"(Sag(?!.*REFORMAT).*CUBE.*FL|SAG.*FL.*CUBE|FLAIR ?WAND)"),
    # R alternate: "AX.*(CUBE.*FL|FL.*CUBE)"
    ("FLAIR",  r"^(?!(SAG|COR).*REFORMAT C).*FLAIR"),   # only take Ax CUBE reformats

    ("WandT2", r"^(?!.*(\+C|POST)).*(WAND.*T2|T2.*WAND)"),
    ("AxT2",   r"^(?!.*STAR)(AX.*T2|T2.*AX)"),
    # R alternate: SagCorT2 = "(^SAG.*T2.*[ABC]?$|^COR.*T2.*[ABC]?$)"
    ("OtherT2", r"(FSE.*T2|T2.*FSE|T2.*STEALTH)"),

    ("DSC",    r"DSC"),
    ("DCE",    r"DCE"),
    ("Trace",  r"(T2|D[WT]I).*TRACE"),
    ("eADC",   r"^Exponen"),
    ("ADC",    r"(^APPAR|I_ADC)"),
    ("AvgDC",  r"(Average DC)"),
    ("DWIDTI", r"(DWI$|^(Ax )?DTI|DIFFUSION |Ax DT1)"),
    ("FA",     r"(FRACT|FA |ANISO)"),
]

UNCLASSIFIED = "unclassified"
TYPE_NAMES = [name for name, _ in SERIES_TYPES]

# Columns of series_index.csv this script needs.
DESC_COL = "SeriesDescription"
PROTO_COL = "ProtocolName"
STUDY_COL = "StudyInstanceUID"


def compile_rules(case_sensitive: bool):
    flags = 0 if case_sensitive else re.IGNORECASE
    out = []
    for name, pattern in SERIES_TYPES:
        try:
            out.append((name, re.compile(pattern, flags)))
        except re.error as exc:
            sys.exit(f"[error] rule {name!r} is not a valid Python regex: {exc}")
    return out


def classify(text: str, rules, all_matches: bool = False):
    """First matching rule wins. Returns (type, [every type that matched])."""
    hits = []
    for name, rx in rules:
        if rx.search(text):
            hits.append(name)
            if not all_matches:
                break
    return (hits[0] if hits else UNCLASSIFIED), hits


def run(args) -> int:
    if not os.path.isfile(args.series_csv):
        sys.exit(f"[error] series database not found: {args.series_csv!r} "
                 f"(produce it with query_series_descriptions.py)")

    df = pd.read_csv(args.series_csv, dtype=str, keep_default_na=False)
    for col in (args.id_col, DESC_COL):
        if col not in df.columns:
            sys.exit(f"[error] column {col!r} not in {args.series_csv} "
                     f"(have: {list(df.columns)})")
    if df.empty:
        sys.exit(f"[error] {args.series_csv} has no rows")

    rules = compile_rules(args.case_sensitive)

    # Match on SeriesDescription; fall back to ProtocolName when the PACS
    # returned no description for a series (an optional C-FIND return key).
    n_fallback = 0
    texts = []
    for _, row in df.iterrows():
        text = str(row[DESC_COL]).strip()
        if not text and not args.no_protocol_fallback and PROTO_COL in df.columns:
            text = str(row[PROTO_COL]).strip()
            n_fallback += bool(text)
        texts.append(text)

    types, all_hits = [], []
    for text in texts:
        t, hits = classify(text, rules, args.all_matches)
        types.append(t)
        all_hits.append(";".join(hits))

    df["matchedOn"] = texts
    df["seriesType"] = types
    if args.all_matches:
        df["allTypes"] = all_hits

    if args.drop_nonanat:
        n_drop = int((df["seriesType"] == "nonAnat").sum())
        df = df[df["seriesType"] != "nonAnat"].copy()
        print(f"[info] --drop-nonanat: removed {n_drop} non-anatomical series")

    # --- per-study counts, one column per type -------------------------------
    # Studies are identified by (AnonymizationID, StudyInstanceUID) but the
    # counts file stays UID-free: a per-patient study ordinal replaces the UID.
    has_study = STUDY_COL in df.columns
    keys = [args.id_col] + ([STUDY_COL] if has_study else [])
    present = [t for t in TYPE_NAMES if t in set(df["seriesType"])]
    columns = present + ([UNCLASSIFIED] if UNCLASSIFIED in set(df["seriesType"]) else [])

    counts = (df.groupby(keys + ["seriesType"]).size().unstack("seriesType", fill_value=0)
                .reindex(columns=columns, fill_value=0).reset_index())
    if has_study:
        counts.insert(1, "StudyIndex",
                      counts.groupby(args.id_col).cumcount() + 1)
        counts = counts.drop(columns=[STUDY_COL])
    counts.insert(len(keys) if has_study else 1, "n_series",
                  counts[columns].sum(axis=1))
    counts = counts.sort_values(args.id_col)

    df.to_csv(args.out, index=False)
    counts.to_csv(args.counts, index=False)

    # --- console report (PHI-free) -------------------------------------------
    n_studies = len(counts)
    mode = "case-sensitive" if args.case_sensitive else "case-insensitive"
    print(f"[info] {len(df)} series from {n_studies} study(ies), "
          f"matched on {DESC_COL} ({mode})"
          + (f"; {n_fallback} fell back to {PROTO_COL}" if n_fallback else ""))

    width = max(len(c) for c in columns) if columns else 12
    for t in columns:
        n_ser = int(counts[t].sum())
        n_std = int((counts[t] > 0).sum())
        print(f"[type] {t:<{width}}  {n_ser:>5} series  {n_std:>4} study(ies)")

    # Studies missing a type are the actionable gap (e.g. no post-contrast T1).
    for t in columns:
        if t in (UNCLASSIFIED, "nonAnat"):
            continue
        missing = int((counts[t] == 0).sum())
        if missing:
            print(f"[gap ] {t:<{width}}  absent in {missing}/{n_studies} study(ies)")

    unmatched = df[df["seriesType"] == UNCLASSIFIED]["matchedOn"]
    if len(unmatched) and args.show_unmatched:
        top = Counter(d or "(no description)" for d in unmatched)
        print(f"[warn] {len(unmatched)} series matched no rule; "
              f"top {min(args.show_unmatched, len(top))} description(s):")
        for desc, n in top.most_common(args.show_unmatched):
            print(f"[warn]   {n:>4}  {desc}")

    print(f"[done] out={args.out}  counts={args.counts}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series_csv", nargs="?", default="series_index.csv",
                    help="series database from query_series_descriptions.py "
                         "(default: series_index.csv)")
    ap.add_argument("--out", default="series_types.csv",
                    help="per-series labels (CONTAINS UIDs; keep out of git)")
    ap.add_argument("--counts", default="series_type_counts.csv",
                    help="per-study type counts (UID-free)")
    ap.add_argument("--id-col", default="AnonymizationID",
                    help="PHI-free patient key column")
    ap.add_argument("--case-sensitive", action="store_true",
                    help="match case-sensitively (literal R grepl(perl=TRUE) "
                         "semantics); default is case-insensitive")
    ap.add_argument("--all-matches", action="store_true",
                    help="also record every rule a series matches in an "
                         "allTypes column (seriesType stays the first match)")
    ap.add_argument("--drop-nonanat", action="store_true",
                    help="exclude series classified as nonAnat from the outputs")
    ap.add_argument("--no-protocol-fallback", action="store_true",
                    help="do not fall back to ProtocolName when a series has no "
                         "SeriesDescription")
    ap.add_argument("--show-unmatched", type=int, default=20, metavar="N",
                    help="list the N most common unclassified descriptions "
                         "(0 to suppress; default 20)")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
