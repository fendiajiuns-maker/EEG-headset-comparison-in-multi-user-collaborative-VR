# Analysis pipeline

## Devices

Two consumer EEG headsets, worn by the same participants in an alternating
paired scheme. One records four channels (TP9, AF7, AF8, TP10); the other
records a single frontal channel.

## Feature set

Band powers per second, log-transformed, then z-scored per participant.

For the four-channel headset the feature vector is 20 band powers (five bands
across four channels) plus frontal alpha and theta asymmetry. A reduced
scenario using only the temporal channels TP9 and TP10 is available through
the `SKENARIO` constant, to test how much frontal EMG contamination
contributes.

Columns `Average_*` and `Raw_*` are excluded, because they are linear
combinations of the band powers already in the vector.

## Windowing

Sliding windows per (participant, condition) block, producing tensors of
shape (N, T, F). Windows are cut separately inside each split so that no
window straddles a train/test boundary.

## Leakage controls

**Cross-subject.** GroupKFold / leave-one-subject-out on participant, so a
test participant contributes nothing to training.

**Within-subject.** Each block (participant x VR version x collaboration x
activity) is split temporally: the first 70 % of rows train, the remainder
tests. Per-participant z-score statistics are computed from that
participant's training rows only, then applied to both splits, so test-set
statistics never reach the normaliser.

**Eye-state control.** Same GroupKFold protocol as cross-subject, on the
eyes-closed versus eyes-open contrast.

## Architectures

CNN, LSTM and CNN-LSTM, trained under identical settings: same windows, same
normalisation, same split seed, same epoch count, batch size, learning rate,
class weights and callbacks. Only the `build_*` function differs.

## Metrics

Balanced accuracy, F1-macro and ROC-AUC, each with a permutation test. Device
differences were tested on the 56 paired participants with two-sided Wilcoxon
signed-rank tests on per-participant balanced accuracy. Confidence intervals
for mean differences came from 10,000 bootstrap resamples.

## Labels

The attention and stress labels come from the device's own eSense indices over
the task sequence. SSQ and FMS questionnaire scores are **not** used as class
labels: they are collected once per session, not per second, and their time
resolution cannot support a per-window label.

## Quality control

Cells are rejected when the eSense value is zero, when a value falls outside
the 0-100 range, or when a run of three or more identical rows indicates a
frozen feed. A cell is retained only if at least 20 seconds of valid data
remain.
