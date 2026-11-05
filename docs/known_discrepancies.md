# Known discrepancies between the manuscript and the published data

`reproduce/reproduce_published_numbers.py` checks 48 reported quantities
against the published CSVs. Forty-six agree. The three entries below do not,
and they are recorded here rather than silently reconciled.

All three concern the self-report section, not the classification results.

## 1. Post-session SSQ totals differ by about 1 point

| | Manuscript | Recomputed |
|---|---|---|
| Four-channel | 24.68 | 25.39 |
| Single-channel | 11.26 | 12.35 |
| p (Wilcoxon, paired) | 0.006 | 0.004 |
| r | 0.292 | 0.286 |

The direction, the magnitude of the gap and the significance all hold. The
manuscript figures were computed on the cohort before two participants under
the age of 18 were excluded under the ethics protocol; the published CSVs
reflect the cohort after exclusion.

**Action:** update the two means in the manuscript to 25.39 and 12.35, and the
test statistics to p = 0.004, r = 0.286.

## 2. The pre-session SSQ difference no longer holds

| | Manuscript | Recomputed |
|---|---|---|
| Four-channel | 6.23 | 5.86 |
| Single-channel | 3.57 | 5.10 |
| p (Wilcoxon, paired) | 0.044 | 0.319 |

This is substantive rather than cosmetic. The manuscript states that
pre-session SSQ already differed between devices and concludes that part of
the post-session gap preceded the session. On the published data that
difference is not significant, and the single-channel pre-session mean is
close to the four-channel one.

**Action:** the sentence beginning "Pre-session SSQ already differed" must be
rewritten. On the current data the honest reading is the opposite: the two
groups started from a comparable baseline, which **strengthens** the
attribution of the post-session gap to the session itself rather than
weakening it. Do not simply drop the sentence, since a reviewer comparing the
manuscript against this repository will look for it.

## 3. Number of sessions per participant

`06_selfreport_ssq_fms_sam.csv` holds 211 rows for 58 participants. Ninety-five
participant-device combinations have two questionnaire sessions, five have
three, and six have one. The reproduction script averages sessions within
participant before testing. The manuscript does not state which rule it used.

**Action:** state the rule explicitly in the Methods section, so that the
reported means are reproducible from the archive without guessing.

## Not a discrepancy

Classification results, paired device differences, participant counts and
model cost figures all reproduce exactly. The nine paired comparisons are
significant at p < 1e-4 in every paradigm and architecture.
