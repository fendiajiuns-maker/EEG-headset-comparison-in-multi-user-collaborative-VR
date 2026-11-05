# English translations of the script headers

The analysis scripts in `code/` carry docstrings and inline comments in
Indonesian, the working language of the project. The scripts are published
unmodified, so that the released code is the code that produced the reported
numbers. This file translates each script's header so that the design of the
analysis is legible without reading Indonesian.

A note on the filenames: the scripts whose names contain `muse` are the
**four-channel** ones (Muse 2, channels TP9, AF7, AF8, TP10). The others are
the **single-channel** ones. The filenames do not say so, so the headings
below state which device each script belongs to.

These are translations, not substitutes. Where a translation and a script
disagree, the script governs.

---

## `04_klasifikasi_cnn_lstm.py` — cross-subject, single-channel device

Classification of ATTENTION versus STRESS from EEG signals
(`MASTER_GABUNGAN.csv`) with three deep-learning architectures: CNN, LSTM and
CNN-LSTM.

Procedure:

1. Load the data, keep only the attention and stress labels.
2. Log-transform the band powers, then z-score PER PARTICIPANT.
3. Sliding-window per (participant, condition) into a tensor of shape (N, T, F).
4. Cross-validation PER PARTICIPANT (GroupKFold / leave-one-subject-out), so
   there is no subject leakage.
5. Train all three models under IDENTICAL settings (apple-to-apple), 100 epochs.
6. Metrics: balanced accuracy, F1-macro, ROC-AUC, each with a permutation test.
7. Write score tables, analysis plots and a Pareto plot to a timestamped folder.

**Apple-to-apple principle.** Everything is held constant except the
architecture: windowing, normalisation, split (same seed), `EPOCHS`, `BATCH`,
`LEARNING_RATE`, `class_weight` and callbacks. Only the `build_*` functions
differ.

Usage:

1. `pip install tensorflow scikit-learn pandas numpy matplotlib seaborn`
   (on CPU: `pip install tensorflow-cpu`)
2. Adjust the section headed "PENGATURAN YANG BOLEH DISESUAIKAN"
   (settings that may be adjusted).
3. `python 04_klasifikasi_cnn_lstm.py`

---

## `04_muse_klasifikasi.py` — cross-subject, four-channel device

Classification of ATTENTION versus STRESS from Muse 2 EEG with
CNN / LSTM / CNN-LSTM, evaluated per subject (GroupKFold / leave-one-subject-out).
Progress is visible per fold and per epoch.

Feature `SKENARIO` (toggle):

- `"SEMUA"` (all) — 20 band powers across four channels, plus frontal
  asymmetry `FAA_alpha` and `FAA_theta`
- `"TANPA_FRONTAL"` (without frontal) — temporal channels TP9 and TP10 only,
  to test the effect of frontal EMG contamination

Run twice, changing `SKENARIO`, to compare.

Metrics: balanced accuracy, F1-macro, accuracy, AUC, each with a permutation
test. Apple-to-apple: everything identical except the architecture.

Usage: `pip install tensorflow scikit-learn pandas numpy matplotlib`, adjust
`BASE_DIR`, set `SKENARIO` and `QUICK_TEST`, then
`python 04_muse_klasifikasi.py`.

---

## `04b_klasifikasi_within_subject.py` — within-subject, single-channel device

Classification of ATTENTION versus STRESS under the WITHIN-SUBJECT paradigm
(per-user calibration). Companion to script 04 (cross-subject / LOSO).

The fundamental difference from script 04:

- Script 04 tests a subject held out entirely, measuring generalisation to a
  NEW PERSON.
- Script 04b lets every subject contribute CALIBRATION data to training, and
  tests on a LATER TIME segment belonging to that same subject.

Leakage-proof design:

1. For each block (participant x VR version x collaboration x activity), rows
   are split TEMPORALLY: the first `TRAIN_RATIO` trains, the remainder tests.
   Windows are cut SEPARATELY within each part, so no window straddles the
   boundary.
2. The per-participant z-score statistics are computed ONLY from that
   subject's TRAINING rows, then applied to both that subject's training and
   test rows, so test statistics never leak into the normaliser.
3. One model is trained on the pooled training rows of all subjects, and
   tested on the pooled test rows.
4. Metrics are computed PER SUBJECT (giving a distribution), plus an aggregate,
   plus a permutation test.

Apple-to-apple: windowing, normalisation, split, epochs, batch, learning rate
and class weights are IDENTICAL; only the architecture (CNN / LSTM / CNN-LSTM)
differs.

Usage: `pip install tensorflow scikit-learn pandas numpy matplotlib`, adjust
the "PENGATURAN" (settings) section, then
`python 04b_klasifikasi_within_subject.py`.

---

## `04b_muse_klasifikasi.py` — within-subject, four-channel device

Classification of ATTENTION versus STRESS from Muse 2 EEG under the
within-subject paradigm: each participant is split TEMPORALLY (first 70 %
trains, last 30 % tests), and the normalisation is computed ONLY from the
training part, which is what makes it leakage-proof. Progress is visible per
model and per epoch. Equivalent to script 04b for the single-channel
device, adapted for the Muse.

Feature `SKENARIO` (toggle): `"SEMUA"` (20 band powers plus frontal asymmetry)
or `"TANPA_FRONTAL"` (TP9 and TP10 only).

Metrics: balanced accuracy, F1-macro, accuracy, AUC, each with a permutation
test.

Usage: `pip install tensorflow scikit-learn pandas numpy matplotlib`, set
`SKENARIO` and `QUICK_TEST`, then `python 04b_muse_klasifikasi.py`.

---

## `04c_klasifikasi_task_baseline.py` — positive control, single-channel device

POSITIVE CONTROL: classification of a contrast the hardware is PROVEN to
capture. Companion to script 04 (attention/stress, cross-subject) and script
04b (within-subject).

Background: the feasibility test showed significant alpha reactivity (eyes
closed greater than eyes open) in about 97 % of participants (r = 0.85). If
the SAME pipeline succeeds on this contrast, that demonstrates the code and
the method are sound, and a failure on attention/stress is then a property of
the signal rather than a broken pipeline.

`TARGET` (chosen in the settings section):

- `"MATA"` (eyes) — eyes closed (1) versus eyes open (0); the main positive
  control
- `"TASK_BASELINE"` — learning or quiz task (1) versus the eye baseline (0)

Evaluation: cross-subject (GroupKFold / leave-one-subject-out), as strict as
script 04. Apple-to-apple: windowing, normalisation, split, epochs, batch,
learning rate and class weights are identical; only the architecture differs.

Usage: `pip install tensorflow scikit-learn pandas numpy matplotlib`, adjust
the settings (`TARGET` above all), then
`python 04c_klasifikasi_task_baseline.py`.

---

## `04c_muse_klasifikasi.py` — positive control, four-channel device

Classification of the physiological eyes-CLOSED versus eyes-OPEN contrast
(the Berger effect) from Muse 2 EEG, cross-subject. A positive control: if it
succeeds, the pipeline is sound. Equivalent to script 04c for the single-channel
device, adapted for the Muse.

Feature `SKENARIO` (toggle): `"SEMUA"` (20 band powers plus frontal asymmetry)
or `"TANPA_FRONTAL"` (TP9 and TP10 only).

Metrics: balanced accuracy, F1-macro, accuracy, AUC, each with a permutation
test.

Usage: `pip install tensorflow scikit-learn pandas numpy matplotlib`, set
`SKENARIO` and `QUICK_TEST`, then `python 04c_muse_klasifikasi.py`.

---

## Glossary of Indonesian terms appearing in the code

| Indonesian | English |
|---|---|
| peserta | participant |
| latih / uji | train / test |
| daya pita | band power |
| kanal | channel |
| jendela | window |
| bocor / anti-bocor | leak / leakage-proof |
| pengaturan | settings |
| kondisi | condition |
| aktivitas | activity |
| kolaborasi | collaboration |
| versi_vr | VR version |
| mata tertutup / terbuka | eyes closed / open |
| atensi / stres | attention / stress |
| uji permutasi | permutation test |
| hasil | result |
