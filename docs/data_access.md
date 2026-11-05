# Obtaining the data

## Derived dataset (public)

The derived dataset is archived on Zenodo under DOI
`10.5281/zenodo.XXXXXXX`. Download the archive, unpack it, and copy its
`data/` folder into the root of this repository:

```
eeg-vr-device-comparison/
  data/
    00_cohort_summary.csv
    01_participants.csv
    02_classification_cross_subject.csv
    03_classification_within_subject.csv
    04_classification_eye_state_control.csv
    05_paired_device_differences.csv
    06_selfreport_ssq_fms_sam.csv
    07_model_cost.csv
```

`reproduce/reproduce_published_numbers.py` then runs with no arguments. The
folder is listed in `.gitignore`, so the data is never committed here.

## What is in the archive

| File | Rows | Contents |
|---|---|---|
| `00_cohort_summary.csv` | 15 | aggregate participant characteristics, counts only |
| `01_participants.csv` | 58 | anonymous code, sex, two-level age band |
| `02_classification_cross_subject.csv` | 354 | balanced accuracy, F1-macro and AUC per participant, device and architecture |
| `03_classification_within_subject.csv` | 354 | balanced accuracy per participant, device and architecture |
| `04_classification_eye_state_control.csv` | 354 | positive-control results, same shape as above |
| `05_paired_device_differences.csv` | 504 | per-participant four-channel minus single-channel difference |
| `06_selfreport_ssq_fms_sam.csv` | 211 | SSQ subscales before and after, peak and mean FMS, SAM valence and arousal |
| `07_model_cost.csv` | 3 | parameter count and inference latency per architecture |

## What is not released, and why

**Raw EEG recordings.** Per-second records of brain activity are not
distributed. Band-power time series over a known task sequence carry
re-identification risk, and the consent obtained from participants covers use
within this research rather than open-ended redistribution.

**The code-to-participant key.** Anonymous codes `S001`–`S063` were assigned
by a seeded random shuffle, so they do not follow recruitment order. The
mapping from code to participant is held by the corresponding author and is
not part of either the Zenodo archive or this repository.

**Individual participant characteristics beyond sex and age band.** An audit
found nine participants whose combination of six recorded characteristics was
unique in the cohort, which would have allowed re-identification by anyone
holding one of those characteristics. Prior VR experience, corrected vision,
motion sickness history and balance disorder are therefore released only as
aggregate counts in `00_cohort_summary.csv`. The released participant table
reaches a minimum cell size of three with no unique combinations.

## Requests for the raw recordings

Researchers who need the per-second records for methodological replication
should contact the corresponding author. Release is subject to approval by the
institutional ethics committee and to a data use agreement.
