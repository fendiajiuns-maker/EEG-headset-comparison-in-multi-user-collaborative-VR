# EEG headset comparison in multi-user collaborative VR

Analysis code and reproducibility material for the study comparing a
four-channel consumer EEG headset against a single-channel consumer EEG
headset during collaborative learning tasks in a multi-user virtual reality
environment.

This repository contains **code only**. The derived dataset the code was run
on is archived separately (see [Data](#data)). Raw EEG recordings are not
released.

## What the study did

Fifty-eight participants completed learning and quiz tasks inside a
multi-user VR environment. Each participant wore both headsets in an
alternating paired scheme, so device effects are measured within participant
rather than between groups. Three deep-learning architectures (CNN, LSTM and
CNN-LSTM) were trained under identical settings on band-power features, and
evaluated under three paradigms:

| Paradigm | What it tests | Split |
|---|---|---|
| Cross-subject | generalisation to an unseen person | GroupKFold / LOSO |
| Within-subject | per-user calibration | temporal, 70 % train / 30 % test |
| Eye-state control | whether the pipeline itself works | GroupKFold / LOSO |

The eye-state contrast (eyes closed versus open, the Berger effect) is a
positive control. If the same pipeline classifies a contrast the hardware is
known to capture, then a failure on the attention/stress contrast is a
property of the signal, not a bug in the code.

## Repository layout

```
code/                             analysis scripts, exactly as run
  04_klasifikasi_cnn_lstm.py      cross-subject, four-channel headset
  04_muse_klasifikasi.py          cross-subject, single-channel headset
  04b_klasifikasi_within_subject.py   within-subject, four-channel
  04b_muse_klasifikasi.py             within-subject, single-channel
  04c_klasifikasi_task_baseline.py    eye-state control, four-channel
  04c_muse_klasifikasi.py             eye-state control, single-channel
reproduce/
  reproduce_published_numbers.py  recomputes every reported number from the
                                  published CSVs and prints it next to the
                                  value in the manuscript
docs/
  pipeline.md                     preprocessing, feature set, leakage controls
  data_access.md                  how to obtain the dataset
  known_discrepancies.md          numbers that do not yet agree, and why
```

## Quick start: verify the reported numbers

This needs neither a GPU nor TensorFlow nor the raw recordings. It takes a
few seconds.

```bash
git clone https://github.com/<username>/eeg-vr-device-comparison.git
cd eeg-vr-device-comparison
pip install pandas numpy scipy

# download the dataset archive from Zenodo, unpack it, and copy its data/
# folder to ./data  (see docs/data_access.md)

python reproduce/reproduce_published_numbers.py
```

Each line prints the recomputed value beside the published value, marked
`ok`, `BEDA` (differs) or `TANDAI` (flagged for review). The script exits
non-zero if anything differs, so it can be wired into CI.

## Rerunning the full training

The six scripts in `code/` are the scripts as they were run, including the
Windows paths they were run with. Each script has a `BASE_DIR` constant near
the top that must be changed to the folder holding the merged per-second
EEG records. See `code/README.md`.

Training needs TensorFlow and takes hours on CPU:

```bash
pip install -r requirements.txt
python code/04_klasifikasi_cnn_lstm.py
```

Per-participant results are written to a timestamped folder. The published
CSVs in the dataset archive are those results, concatenated and
de-identified.

## Data

The derived dataset is archived on Zenodo:

- DOI: `10.5281/zenodo.XXXXXXX` (replace once minted)
- Contents: per-participant classification results for all three paradigms,
  paired device differences, self-report SSQ/FMS/SAM scores, model cost
  figures, a data dictionary and provenance notes.

Participants appear only as randomly assigned anonymous codes (`S001`–`S063`).
The code-to-participant key is held by the corresponding author and is not
distributed. Participant characteristics are released in aggregate form to
prevent re-identification. See `docs/data_access.md`.

## Citation

See `CITATION.cff`. Please cite both the article and the Zenodo archive.

## Licence

Code is released under the MIT Licence (`LICENSE`). The dataset in the Zenodo
archive carries its own licence, stated in that archive.
