# -*- coding: utf-8 -*-
"""
reproduce_published_numbers.py
==============================================================
Recomputes every number reported in the manuscript from the derived CSV files
archived on Zenodo, and prints each recomputed value next to the value printed
in the manuscript, so that any disagreement is immediately visible.

This script trains nothing. It needs neither TensorFlow nor the raw EEG
recordings. It verifies only that the numbers in the manuscript really do
follow from the published result files.

Usage
-----
    pip install pandas numpy scipy
    python reproduce/reproduce_published_numbers.py

By default the script looks for the data in ./data relative to the repository
root. Download the Zenodo archive, unpack it, and place its data folder like
this:

    eeg-vr-device-comparison/
      data/                      <- contents of data/ from the Zenodo archive
        00_cohort_summary.csv
        01_participants.csv
        02_classification_cross_subject.csv
        03_classification_within_subject.csv
        04_classification_eye_state_control.csv
        05_paired_device_differences.csv
        06_selfreport_ssq_fms_sam.csv
        07_model_cost.csv
      reproduce/
        reproduce_published_numbers.py

Or point the script at the folder directly:

    python reproduce/reproduce_published_numbers.py --data /path/to/data

The script exits with status 1 if any checked value disagrees, so it can be
wired into continuous integration.
==============================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Where the data lives. The repository root is the parent of this script's
# folder.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", default=str(ROOT / "data"),
                help="folder holding the CSV files from the Zenodo archive")
arg = ap.parse_args()
DATA = Path(arg.data)

if not DATA.is_dir():
    sys.exit(f"Data folder not found: {DATA}\n"
             f"Download the Zenodo archive, then rerun with --data <folder>.")

# ---------------------------------------------------------------------------
# Tally of agreements. Every check prints one line.
# ---------------------------------------------------------------------------
agree = 0
differ = 0
flagged = 0


def check(label, computed, reported, tol=0.05, unit=""):
    """Compare one recomputed value against the value printed in the paper."""
    global agree, differ
    gap = abs(computed - reported)
    status = "ok    " if gap <= tol else "DIFFER"
    if gap <= tol:
        agree += 1
    else:
        differ += 1
    print(f"  {status} {label:<44s} computed={computed:>8.2f}{unit}"
          f"  paper={reported:>8.2f}{unit}")


def flag(label, body):
    """Record a value deliberately left uncompared, with the reason."""
    global flagged
    flagged += 1
    print(f"  FLAG   {label}")
    for line in body.splitlines():
        print(f"         {line}")


def load(name):
    """Read one CSV file from the data folder."""
    p = DATA / name
    if not p.exists():
        sys.exit(f"Missing file: {p}")
    return pd.read_csv(p)


print("=" * 78)
print("VERIFICATION OF REPORTED NUMBERS AGAINST THE PUBLISHED DATA")
print(f"data folder: {DATA}")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. Cohort
# ---------------------------------------------------------------------------
print("\n1. COHORT")
participants = load("01_participants.csv")
paired = load("05_paired_device_differences.csv")

check("participants analysed", len(participants), 58, tol=0)
check("participants with both devices",
      paired.participant_id.nunique(), 56, tol=0)

# ---------------------------------------------------------------------------
# 2. Balanced accuracy across the three evaluation paradigms.
#    Reported values: (CNN, CNN-LSTM, LSTM) per device, in per cent.
# ---------------------------------------------------------------------------
REPORTED_ACCURACY = {
    "02_classification_cross_subject.csv": {
        "label": "cross-subject",
        "four_channel": {"CNN": 59.0, "CNN-LSTM": 60.9, "LSTM": 61.8},
        "single_channel": {"CNN": 49.7, "CNN-LSTM": 49.8, "LSTM": 49.7},
    },
    "03_classification_within_subject.csv": {
        "label": "within-subject",
        "four_channel": {"CNN": 61.3, "CNN-LSTM": 65.7, "LSTM": 66.0},
        "single_channel": {"CNN": 49.5, "CNN-LSTM": 50.4, "LSTM": 53.4},
    },
    "04_classification_eye_state_control.csv": {
        "label": "eye-state control",
        "four_channel": {"CNN": 88.9, "CNN-LSTM": 90.0, "LSTM": 90.2},
        "single_channel": {"CNN": 71.8, "CNN-LSTM": 71.6, "LSTM": 69.1},
    },
}

DEVICE_NAME = {"four_channel": "four-channel", "single_channel": "single-channel"}

print("\n2. BALANCED ACCURACY (%)")
for fname, ref in REPORTED_ACCURACY.items():
    d = load(fname)
    print(f"\n  -- {ref['label']} --")
    # mean balanced accuracy per (device, architecture), computed from the
    # published per-participant values
    means = d.groupby(["device", "model"])["balanced_accuracy"].mean() * 100
    for device in ("four_channel", "single_channel"):
        for model in ("CNN", "CNN-LSTM", "LSTM"):
            check(f"{DEVICE_NAME[device]} {model}",
                  means.loc[(device, model)],
                  ref[device][model], unit="%")

# ---------------------------------------------------------------------------
# 3. Paired differences between devices.
#    Reported values in percentage points, with the number of participants
#    favouring the four-channel device.
# ---------------------------------------------------------------------------
REPORTED_DIFFERENCE = {
    ("cross_subject", "CNN"): (8.8, 47),
    ("cross_subject", "CNN-LSTM"): (10.8, 46),
    ("cross_subject", "LSTM"): (11.6, 47),
    ("within_subject", "CNN"): (11.9, 40),
    ("within_subject", "CNN-LSTM"): (15.7, 48),
    ("within_subject", "LSTM"): (12.8, 42),
    ("eye_state_control", "CNN"): (16.8, 49),
    ("eye_state_control", "CNN-LSTM"): (18.2, 49),
    ("eye_state_control", "LSTM"): (20.9, 51),
}

print("\n3. PAIRED DIFFERENCE, FOUR-CHANNEL MINUS SINGLE-CHANNEL (pp)")
for (task, model), (diff_paper, favour_paper) in REPORTED_DIFFERENCE.items():
    g = paired[(paired.task == task) & (paired.model == model)]
    check(f"{task} {model} difference", g.difference.mean() * 100,
          diff_paper, unit=" pp")
    check(f"{task} {model} favour four-channel",
          (g.difference > 0).sum(), favour_paper, tol=0)

# two-sided paired Wilcoxon tests, as stated in the manuscript
print("\n   two-sided paired Wilcoxon tests (p):")
for (task, model) in REPORTED_DIFFERENCE:
    g = paired[(paired.task == task) & (paired.model == model)]
    w = stats.wilcoxon(g.four_channel, g.single_channel)
    print(f"     {task:<18s} {model:<9s} n={len(g):>3d}  p={w.pvalue:.2e}")

# ---------------------------------------------------------------------------
# 4. Model cost
# ---------------------------------------------------------------------------
print("\n4. MODEL COST")
cost = load("07_model_cost.csv").set_index("model")
REPORTED_COST = {
    "CNN": (18586, 1.14),
    "LSTM": (35777, 1.69),
    "CNN-LSTM": (11681, 4.01),
}
for model, (params, latency) in REPORTED_COST.items():
    check(f"{model} parameter count", cost.loc[model, "parameters"],
          params, tol=0)
    check(f"{model} inference latency",
          cost.loc[model, "inference_latency_ms"], latency,
          tol=0.005, unit=" ms")

# ---------------------------------------------------------------------------
# 5. Self-report: SSQ and FMS
#    Note: file 06 holds more than one session for most participants
#    (95 participant-device combinations with 2 sessions, 5 with 3, 6 with 1),
#    so sessions are averaged within participant before testing.
# ---------------------------------------------------------------------------
print("\n5. SELF-REPORT, SSQ AND FMS")
self_report = load("06_selfreport_ssq_fms_sam.csv")
codes56 = set(paired.participant_id)
sr56 = self_report[self_report.participant_id.isin(codes56)]


def paired_test(d, column):
    """Average sessions within participant, then run a paired Wilcoxon test."""
    wide = d.pivot_table(index="participant_id", columns="device",
                         values=column).dropna()
    w = stats.wilcoxon(wide.four_channel, wide.single_channel)
    z = stats.norm.isf(w.pvalue / 2)          # effect size r = z / sqrt(2n)
    r = z / np.sqrt(2 * len(wide))
    return (wide.four_channel.mean(), wide.single_channel.mean(),
            w.pvalue, r, len(wide))


m4, m1, p, r, n = paired_test(sr56, "ssq_total_after")
print(f"  SSQ post-session, 56 paired participants: "
      f"four-channel={m4:.2f}  single-channel={m1:.2f}  n={n}  "
      f"p={p:.3f}  r={r:.3f}")
check("SSQ post-session four-channel", m4, 24.68, tol=0.5)
check("SSQ post-session single-channel", m1, 11.26, tol=0.5)
check("SSQ post-session p value", p, 0.006, tol=0.005)

m4b, m1b, pb, rb, nb = paired_test(sr56, "ssq_total_before")
print(f"  SSQ pre-session, 56 paired participants:  "
      f"four-channel={m4b:.2f}  single-channel={m1b:.2f}  n={nb}  p={pb:.3f}")
flag("SSQ pre-session",
     f"The manuscript reports 6.23 versus 3.57 with p = 0.044, and concludes\n"
     f"that part of the gap preceded the session. On the published data the\n"
     f"values are {m4b:.2f} versus {m1b:.2f} with p = {pb:.3f}, which is not\n"
     f"significant. That sentence must be revised before resubmission.\n"
     f"See docs/known_discrepancies.md.")

m4c, m1c, pc, rc, nc = paired_test(sr56, "fms_peak")
print(f"  Peak FMS, 56 paired participants:         "
      f"four-channel={m4c:.2f}  single-channel={m1c:.2f}  n={nc}  p={pc:.3f}")
check("peak FMS not significant", 1.0 if pc > 0.05 else 0.0, 1.0, tol=0)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"SUMMARY: {agree} agree, {differ} differ, {flagged} flagged for review")
print("=" * 78)
sys.exit(0 if differ == 0 else 1)
