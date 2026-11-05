# Running the analysis scripts

The six scripts in this folder are the scripts **as they were run** during the
study, including the Windows paths they were run with. They have not been
tidied up, so that the published code is genuinely the code that produced the
numbers in the manuscript.

The script docstrings and inline comments are in Indonesian, the working
language of the project. English translations of all six headers are in
`docs/script_headers_en.md`.

## The one thing you must change

Each script has a `BASE_DIR` constant near the top, for example:

```python
BASE_DIR = r"D:\python vscode\venv\14_klasifikasieeg_muse_2026"
```

Change it to the folder holding the merged per-second EEG records
(`MASTER_GABUNGAN.csv` for the four-channel device, its counterpart for the
single-channel device). Nothing else needs changing.

The per-second records are not published. See `docs/data_access.md`.

## Script pairs

| Paradigm | Four-channel (Muse 2) | Single-channel |
|---|---|---|
| Cross-subject | `04_muse_klasifikasi.py` | `04_klasifikasi_cnn_lstm.py` |
| Within-subject | `04b_muse_klasifikasi.py` | `04b_klasifikasi_within_subject.py` |
| Eye-state control | `04c_muse_klasifikasi.py` | `04c_klasifikasi_task_baseline.py` |

The scripts whose names contain `muse` are the four-channel ones: they read
`MASTER_MUSE_bersih.csv` and build features from channels TP9, AF7, AF8 and
TP10. The others are the single-channel ones: they read
`MASTER_EEG_bersih.csv` and use the device's own Attention and Meditation
eSense values alongside the band powers.

Within each pair the windowing, normalisation, split seed, epoch count, batch
size, learning rate, class weights and callbacks are **identical**. Only the
`build_*` functions differ between architectures. That is what makes the
comparison across architectures a fair one.

## Switches inside the scripts

The four-channel scripts carry a `SKENARIO` constant:

- `"SEMUA"` — all 20 band powers across four channels, plus frontal asymmetry
- `"TANPA_FRONTAL"` — temporal channels TP9 and TP10 only, to test how much
  frontal EMG contamination contributes

Run twice with the two values to compare. Script `04c` carries a `TARGET`
constant with the options `"MATA"` (eyes closed versus open, the main positive
control) and `"TASK_BASELINE"` (task versus eye baseline).

A `QUICK_TEST` constant cuts the epoch count for a fast smoke run. Make sure
it is `False` for a real run.

## Run time

On CPU, one cross-subject script with leave-one-subject-out over 59
participants and three architectures takes several hours. Output goes to a
timestamped folder holding per-participant score tables, analysis plots and a
Pareto plot.
