# -*- coding: utf-8 -*-
r"""
04_klasifikasi_cnn_lstm.py
============================================================
Klasifikasi emosi ATENSI vs STRES dari sinyal EEG (MASTER_GABUNGAN.csv)
dengan tiga arsitektur pembelajaran mendalam: CNN, LSTM, CNN-LSTM.

Alur:
  1. Muat data -> ambil hanya label atensi/stres.
  2. Log-transform daya pita -> normalisasi z-score PER-PESERTA.
  3. Windowing (sliding window) per (peserta,kondisi) -> tensor (N, T, F).
  4. Cross-validation PER-PESERTA (GroupKFold / LOSO) -> tak ada kebocoran subjek.
  5. Latih ketiga model dgn pengaturan IDENTIK (apple-to-apple), 100 epoch.
  6. Metrik: balanced accuracy, F1-macro, ROC-AUC + uji permutasi.
  7. Simpan tabel nilai + grafik analisis + grafik Pareto ke folder ber-timestamp.

PRINSIP APPLE-TO-APPLE:
  Semua sama kecuali arsitektur -> window, normalisasi, split (seed sama),
  EPOCHS, BATCH, LEARNING_RATE, class_weight, callback. Hanya build_* berbeda.

Cara pakai:
  1. pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
     (untuk CPU: pip install tensorflow-cpu)
  2. Sesuaikan bagian "PENGATURAN YANG BOLEH DISESUAIKAN".
  3. python 04_klasifikasi_cnn_lstm.py
============================================================
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#  PENGATURAN YANG BOLEH DISESUAIKAN  (ubah di sini saja)
# ============================================================
BASE_DIR   = r"D:\python vscode\venv\12_klasifikasieeg-2026"
FILE_DATA  = os.path.join(BASE_DIR, "data_bersih", "MASTER_EEG_bersih.csv")
DIR_HASIL  = os.path.join(BASE_DIR, "hasil_klasifikasi_61baru")

# --- uji cepat vs run penuh ---
QUICK_TEST = False          # True: GroupKFold 5-fold + 20 epoch (cepat, utk coba).
                           # False: LOSO penuh + 100 epoch (sesuai permintaan; LAMA).

# --- parameter windowing ---
WINDOW      = 10           # panjang jendela (detik) pada data ~1 Hz
OVERLAP     = 0.5          # tumpang tindih antar jendela (0..0.9)

# --- fitur ---
PAKAI_ESENSE   = True      # sertakan Attention & Meditation sbg fitur?
LOG_BAND       = True      # log-transform daya pita sebelum normalisasi?

# --- pelatihan (IDENTIK untuk ketiga model = apple-to-apple) ---
EPOCHS        = 100        # << permintaan: 100 epoch (dipangkas jika QUICK_TEST)
BATCH         = 32
LEARNING_RATE = 1e-3
EARLY_STOP    = False      # False = tampilkan penuh 100 epoch; True = berhenti dini
SEED          = 42

# --- cross validation ---
CV_MODE     = "LOSO"       # "LOSO" (leave-one-subject-out) atau "KFOLD"
N_SPLITS    = 5            # dipakai bila CV_MODE="KFOLD" atau QUICK_TEST

# --- uji permutasi (berbasis prediksi, murah) ---
N_PERMUTASI = 1000

# ============================================================

if QUICK_TEST:
    EPOCHS = 20
    CV_MODE = "KFOLD"

STAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RUN = os.path.join(DIR_HASIL, f"run_{STAMP}")

SIG_BAND = ['Delta','Theta','Low Alpha','High Alpha',
            'Low Beta','High Beta','Low Gamma','Mid Gamma']
ESENSE   = ['Attention','Meditation']
GRUP_KOND = ['partisipan','versi_vr','kolaborasi','aktivitas']

np.random.seed(SEED)


# ============================================================
# 1) MUAT & SIAPKAN FITUR
# ============================================================
def siapkan_data():
    df = pd.read_csv(FILE_DATA)
    # buat label_emosi bila belum ada (file proyek 12 hanya punya 'aktivitas')
    if 'label_emosi' not in df.columns:
        df['label_emosi'] = df['aktivitas'].astype(str).map(
            {'Belajar':'atensi','Quiz':'stres',
             'MataTerbuka':'baseline','MataTertutup':'baseline'}).fillna('lain')
    # hanya kelas atensi vs stres
    df = df[df['label_emosi'].isin(['atensi','stres'])].copy()
    df['y'] = (df['label_emosi'] == 'stres').astype(int)   # atensi=0, stres=1

    fitur = SIG_BAND + (ESENSE if PAKAI_ESENSE else [])
    for c in fitur:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=fitur).reset_index(drop=True)

    # log-transform daya pita (memori: band power sebaiknya di-log)
    if LOG_BAND:
        for c in SIG_BAND:
            df[c] = np.log1p(df[c].clip(lower=0))

    # normalisasi z-score PER-PESERTA (label-agnostik, tanpa kebocoran antar-subjek)
    # pakai transform (mempertahankan kolom & index; aman di pandas 2.x)
    gmean = df.groupby('partisipan')[fitur].transform('mean')
    gstd  = df.groupby('partisipan')[fitur].transform('std').replace(0, np.nan)
    df[fitur] = (df[fitur] - gmean) / (gstd + 1e-8)
    df[fitur] = df[fitur].fillna(0.0)
    return df, fitur


# ============================================================
# 2) WINDOWING -> tensor (N, T, F)
# ============================================================
def buat_jendela(df, fitur):
    step = max(1, int(WINDOW * (1 - OVERLAP)))
    X, y, grp = [], [], []
    for _, sub in df.groupby(GRUP_KOND, dropna=False):
        arr = sub[fitur].to_numpy(np.float32)
        lab = sub['y'].to_numpy()
        pid = sub['partisipan'].iloc[0]
        for s in range(0, len(arr) - WINDOW + 1, step):
            X.append(arr[s:s+WINDOW])
            y.append(int(round(lab[s:s+WINDOW].mean())))  # label mayoritas jendela
            grp.append(pid)
    return np.array(X), np.array(y), np.array(grp)


# ============================================================
# 3) MODEL (IDENTIK regime; hanya arsitektur berbeda)
# ============================================================
def build_cnn(T, F):
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input((T, F)),
        layers.Conv1D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name="CNN")
    return m

def build_lstm(T, F):
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input((T, F)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name="LSTM")
    return m

def build_cnnlstm(T, F):
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input((T, F)),
        layers.Conv1D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name="CNN-LSTM")
    return m

MODELS = {"CNN": build_cnn, "LSTM": build_lstm, "CNN-LSTM": build_cnnlstm}


# ============================================================
# 4) SATU FOLD: latih + prediksi
# ============================================================
def latih_evaluasi(build_fn, Xtr, ytr, Xte, T, F, verbose):
    from tensorflow.keras import optimizers, callbacks
    from sklearn.utils.class_weight import compute_class_weight
    m = build_fn(T, F)
    m.compile(optimizer=optimizers.Adam(LEARNING_RATE),
              loss='binary_crossentropy', metrics=['accuracy'])
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=ytr)
    cbs = []
    if EARLY_STOP:
        cbs.append(callbacks.EarlyStopping(patience=10, restore_best_weights=True))
    t0 = time.time()
    hist = m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH,
                 class_weight={0: cw[0], 1: cw[1]}, verbose=verbose, callbacks=cbs)
    t_latih = time.time() - t0
    # latensi inferensi (ms/jendela)
    t1 = time.time(); prob = m.predict(Xte, verbose=0).ravel()
    t_infer = (time.time() - t1) / max(1, len(Xte)) * 1000
    n_param = int(m.count_params())
    return prob, hist.history, t_latih, t_infer, n_param


# ============================================================
# 5) MAIN
# ============================================================
def main():
    os.makedirs(DIR_RUN, exist_ok=True)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 roc_auc_score, confusion_matrix)

    df, fitur = siapkan_data()
    X, y, grp = buat_jendela(df, fitur)
    T, F = X.shape[1], X.shape[2]
    n_subj = len(np.unique(grp))
    print(f"Jendela: {X.shape} | fitur={fitur}")
    print(f"Subjek: {n_subj} | kelas: atensi={int((y==0).sum())}, stres={int((y==1).sum())}")

    # pemecah CV
    if CV_MODE == "LOSO":
        splitter = LeaveOneGroupOut(); n_fold = n_subj
    else:
        n_fold = min(N_SPLITS, n_subj)
        splitter = GroupKFold(n_splits=n_fold)
    print(f"CV: {CV_MODE} ({n_fold} fold) | epoch={EPOCHS} | quick={QUICK_TEST}\n")

    baris_fold = []          # metrik per fold
    oof = {k: {'yt': [], 'yp': [], 'pr': []} for k in MODELS}   # out-of-fold
    hist_simpan = {k: None for k in MODELS}    # simpan history fold pertama
    lat = {k: {'infer': [], 'param': 0, 'latih': []} for k in MODELS}

    for i, (tr, te) in enumerate(splitter.split(X, y, grp), 1):
        Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
        # lewati fold bila kelas train tak lengkap
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        for nama, build_fn in MODELS.items():
            vb = 1 if (i == 1 and not QUICK_TEST) else 0   # tampilkan epoch fold-1
            if i == 1:
                print(f"--- Fold {i} · {nama} ---")
            prob, hist, tl, ti, npar = latih_evaluasi(
                build_fn, Xtr, ytr, Xte, T, F, vb)
            yp = (prob >= 0.5).astype(int)
            baris_fold.append({'fold': i, 'model': nama,
                'balanced_acc': balanced_accuracy_score(yte, yp),
                'f1_macro': f1_score(yte, yp, average='macro'),
                'auc': roc_auc_score(yte, prob) if len(np.unique(yte))>1 else np.nan,
                'n_test': len(yte), 'waktu_latih_s': round(tl,2)})
            oof[nama]['yt'] += list(yte); oof[nama]['yp'] += list(yp)
            oof[nama]['pr'] += list(prob)
            lat[nama]['infer'].append(ti); lat[nama]['param'] = npar
            lat[nama]['latih'].append(tl)
            if hist_simpan[nama] is None:
                hist_simpan[nama] = hist
        print(f"[fold {i}/{n_fold}] selesai")

    # ---------- tabel per fold + ringkasan ----------
    fold_df = pd.DataFrame(baris_fold)
    fold_df.to_csv(os.path.join(DIR_RUN, "metrik_per_fold.csv"), index=False)

    ring = (fold_df.groupby('model')[['balanced_acc','f1_macro','auc']]
            .agg(['mean','std']).round(4))
    ring.columns = ['_'.join(c) for c in ring.columns]
    ring = ring.reset_index()
    # tambah latensi & param & uji permutasi
    perm_rows = []
    for nama in MODELS:
        yt = np.array(oof[nama]['yt']); yp = np.array(oof[nama]['yp'])
        ba = balanced_accuracy_score(yt, yp)
        # permutasi label vs prediksi tetap (murah, sahih)
        null = []
        rng = np.random.default_rng(SEED)
        for _ in range(N_PERMUTASI):
            null.append(balanced_accuracy_score(rng.permutation(yt), yp))
        null = np.array(null)
        p = (1 + np.sum(null >= ba)) / (N_PERMUTASI + 1)
        perm_rows.append({'model': nama, 'balanced_acc_oof': round(ba,4),
                          'perm_mean': round(null.mean(),4), 'p_value': round(p,4),
                          'latensi_ms': round(np.mean(lat[nama]['infer']),3),
                          'n_param': lat[nama]['param'],
                          'waktu_latih_rata_s': round(np.mean(lat[nama]['latih']),2)})
    perm_df = pd.DataFrame(perm_rows)
    ringkas = ring.merge(perm_df, on='model')
    ringkas.to_csv(os.path.join(DIR_RUN, "ringkasan_model.csv"), index=False)

    print("\n=== RINGKASAN MODEL ===")
    print(ringkas.to_string(index=False))

    # ---------- GRAFIK ----------
    _plot_semua(fold_df, oof, hist_simpan, perm_df, ringkas)

    with open(os.path.join(DIR_RUN, "INFO.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run {STAMP}\nQUICK_TEST={QUICK_TEST}, CV={CV_MODE}, EPOCHS={EPOCHS}\n")
        f.write(f"Jendela={X.shape}, fitur={fitur}\n")
        f.write("Apple-to-apple: window/normalisasi/split/epoch/batch/lr/"
                "class_weight identik; hanya arsitektur berbeda.\n")
    print("\n[SELESAI] Output di:", DIR_RUN)


# ============================================================
# 6) GRAFIK ANALISIS + PARETO
# ============================================================
def _plot_semua(fold_df, oof, hist_simpan, perm_df, ringkas):
    from sklearn.metrics import confusion_matrix
    warna = {'CNN':'#2c6fbb','LSTM':'#c0392b','CNN-LSTM':'#2e8b57'}

    # (a) kurva belajar (loss & acc) fold-1
    fig, ax = plt.subplots(1, 2, figsize=(12,4.5))
    for nama, h in hist_simpan.items():
        if h is None: continue
        ax[0].plot(h['loss'], label=nama, color=warna.get(nama))
        ax[1].plot(h['accuracy'], label=nama, color=warna.get(nama))
    ax[0].set_title("Kurva Loss (fold-1)"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[1].set_title("Kurva Accuracy (fold-1)"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("acc")
    ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"01_kurva_belajar.png"),dpi=150); plt.close()

    # (b) boxplot balanced_acc per fold
    fig, ax = plt.subplots(figsize=(7,5))
    data = [fold_df[fold_df.model==m]['balanced_acc'] for m in warna]
    bp = ax.boxplot(data, labels=list(warna), patch_artist=True)
    for patch, m in zip(bp['boxes'], warna): patch.set_facecolor(warna[m]); patch.set_alpha(0.6)
    ax.axhline(0.5, color='k', ls=':', label='tebakan (0,5)')
    ax.set_ylabel("balanced accuracy"); ax.set_title("Sebaran balanced accuracy antar-fold")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"02_boxplot_balanced_acc.png"),dpi=150); plt.close()

    # (c) confusion matrix per model
    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    for ax, nama in zip(axes, warna):
        cm = confusion_matrix(oof[nama]['yt'], oof[nama]['yp'])
        im = ax.imshow(cm, cmap='Blues')
        for (r,c),v in np.ndenumerate(cm):
            ax.text(c,r,int(v),ha='center',va='center',
                    color='white' if v>cm.max()/2 else 'black')
        ax.set_title(nama); ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['atensi','stres']); ax.set_yticklabels(['atensi','stres'])
        ax.set_xlabel('prediksi'); ax.set_ylabel('aktual')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"03_confusion_matrix.png"),dpi=150); plt.close()

    # (d) uji permutasi (bar balanced_acc vs null)
    fig, ax = plt.subplots(figsize=(7,5))
    x=np.arange(len(perm_df))
    ax.bar(x, perm_df['balanced_acc_oof'], color=[warna[m] for m in perm_df.model], label='observasi')
    ax.plot(x, perm_df['perm_mean'], 'k_', ms=25, label='rata-rata acak')
    ax.axhline(0.5, color='grey', ls=':')
    for i,r in perm_df.iterrows(): ax.text(i, r['balanced_acc_oof']+0.005, f"p={r['p_value']}", ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(perm_df.model); ax.set_ylabel("balanced accuracy")
    ax.set_title("Uji permutasi (OOF)"); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"04_uji_permutasi.png"),dpi=150); plt.close()

    # (e) PARETO benchmarking: akurasi vs latensi (ukuran titik = jumlah parameter)
    fig, ax = plt.subplots(figsize=(8,6))
    xs = ringkas['latensi_ms'].values; ys = ringkas['balanced_acc_oof'].values
    sizes = (ringkas['n_param'].values / ringkas['n_param'].max()) * 900 + 100
    for i,r in ringkas.iterrows():
        ax.scatter(r['latensi_ms'], r['balanced_acc_oof'], s=sizes[i],
                   color=warna[r['model']], alpha=0.7, edgecolor='k', zorder=3)
        ax.annotate(f"{r['model']}\n({r['n_param']:,} param)",
                    (r['latensi_ms'], r['balanced_acc_oof']),
                    xytext=(8,8), textcoords='offset points', fontsize=9)
    # garis frontier Pareto (akurasi tinggi + latensi rendah = kiri-atas)
    pts = ringkas.sort_values('latensi_ms')
    frontier=[]; best=-1
    for _,r in pts.iterrows():
        if r['balanced_acc_oof']>best: frontier.append((r['latensi_ms'],r['balanced_acc_oof'])); best=r['balanced_acc_oof']
    if len(frontier)>1:
        fx,fy=zip(*frontier); ax.plot(fx,fy,'--',color='grey',label='Pareto frontier',zorder=1)
    ax.axhline(0.5,color='grey',ls=':',alpha=0.6,label='tebakan (0,5)')
    ax.set_xlabel("latensi inferensi (ms/jendela) — makin kiri makin cepat")
    ax.set_ylabel("balanced accuracy — makin atas makin akurat")
    ax.set_title("Pareto Benchmarking: Akurasi vs Latensi (ukuran = jumlah parameter)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"05_pareto_benchmarking.png"),dpi=150); plt.close()

    # (f) bar peringkat balanced_acc
    fig, ax = plt.subplots(figsize=(7,5))
    r2 = ringkas.sort_values('balanced_acc_oof', ascending=False)
    ax.bar(r2['model'], r2['balanced_acc_oof'], yerr=r2['balanced_acc_std'],
           color=[warna[m] for m in r2.model], capsize=5)
    ax.axhline(0.5, color='k', ls=':'); ax.set_ylabel("balanced accuracy (OOF)")
    ax.set_title("Peringkat Model"); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"06_peringkat_model.png"),dpi=150); plt.close()


if __name__ == "__main__":
    main()
