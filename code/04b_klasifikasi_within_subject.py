# -*- coding: utf-8 -*-
r"""
04b_klasifikasi_within_subject.py
============================================================
Klasifikasi ATENSI vs STRES paradigma WITHIN-SUBJECT (kalibrasi per-pengguna).
Pelengkap skrip 04 (cross-subject/LOSO).

Perbedaan mendasar dari skrip 04:
  - Skrip 04  : subjek diuji sepenuhnya terpisah (generalisasi ke ORANG BARU).
  - Skrip 04b : tiap subjek menyumbang data KALIBRASI ke latih; diuji pada
                segmen WAKTU BERIKUTNYA milik subjek yang sama.

Desain (anti-bocor):
  1. Untuk tiap blok (peserta x versi_vr x kolaborasi x aktivitas), baris
     dibagi TEMPORAL: TRAIN_RATIO awal -> latih, sisanya -> uji.
     (window dipotong TERPISAH di tiap bagian -> tak ada jendela melintasi batas)
  2. Normalisasi z-score PER-PESERTA dihitung HANYA dari baris LATIH subjek itu,
     lalu diterapkan ke latih & uji subjek tsb (statistik uji tidak bocor).
  3. Satu model dilatih pada gabungan latih semua subjek; diuji pada gabungan uji.
  4. Metrik dihitung PER-SUBJEK (distribusi) + agregat + uji permutasi.

Apple-to-apple: window/normalisasi/split/epoch/batch/lr/class_weight IDENTIK;
hanya arsitektur (CNN/LSTM/CNN-LSTM) yang berbeda.

Cara pakai:
  1. pip install tensorflow scikit-learn pandas numpy matplotlib
  2. Sesuaikan bagian "PENGATURAN".
  3. python 04b_klasifikasi_within_subject.py
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
#  PENGATURAN YANG BOLEH DISESUAIKAN
# ============================================================
BASE_DIR   = r"D:\python vscode\venv\12_klasifikasieeg-2026"
FILE_DATA  = os.path.join(BASE_DIR, "data_bersih", "MASTER_EEG_bersih.csv")
DIR_HASIL  = os.path.join(BASE_DIR, "hasil_klasifikasi_within_61baru")

QUICK_TEST   = False        # True: 20 epoch (cepat). False: 100 epoch (final).
TRAIN_RATIO  = 0.7         # proporsi awal tiap blok untuk latih (sisanya uji)
WINDOW       = 10
OVERLAP      = 0.5
PAKAI_ESENSE = True
LOG_BAND     = True

EPOCHS        = 100        # dipangkas ke 20 bila QUICK_TEST
BATCH         = 32
LEARNING_RATE = 1e-3
SEED          = 42
N_PERMUTASI   = 1000
MIN_WIN_SUBJEK = 5         # min jendela uji per subjek agar metriknya dihitung
# ============================================================

if QUICK_TEST:
    EPOCHS = 20

STAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RUN = os.path.join(DIR_HASIL, f"run_{STAMP}")

SIG_BAND = ['Delta','Theta','Low Alpha','High Alpha',
            'Low Beta','High Beta','Low Gamma','Mid Gamma']
ESENSE   = ['Attention','Meditation']
GRUP_KOND = ['partisipan','versi_vr','kolaborasi','aktivitas']
np.random.seed(SEED)


# ============================================================
# 1) MUAT + SPLIT TEMPORAL + NORMALISASI (dari latih saja)
# ============================================================
def siapkan():
    df = pd.read_csv(FILE_DATA)
    # buat label_emosi bila belum ada (file proyek 12 hanya punya 'aktivitas')
    if 'label_emosi' not in df.columns:
        df['label_emosi'] = df['aktivitas'].astype(str).map(
            {'Belajar':'atensi','Quiz':'stres',
             'MataTerbuka':'baseline','MataTertutup':'baseline'}).fillna('lain')
    df = df[df['label_emosi'].isin(['atensi','stres'])].copy()
    df['y'] = (df['label_emosi'] == 'stres').astype(int)
    fitur = SIG_BAND + (ESENSE if PAKAI_ESENSE else [])
    for c in fitur:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=fitur).reset_index(drop=True)
    df[fitur] = df[fitur].astype('float64')      # pastikan float (hindari error int64)
    if LOG_BAND:
        for c in SIG_BAND:
            df[c] = np.log1p(df[c].clip(lower=0))

    # (1) tandai split temporal per BLOK
    df['split'] = 'train'
    for _, idx in df.groupby(GRUP_KOND, dropna=False).groups.items():
        idx = list(idx)
        n_tr = int(np.ceil(len(idx) * TRAIN_RATIO))
        for i in idx[n_tr:]:
            df.at[i, 'split'] = 'test'

    # (2) normalisasi z-score per-peserta memakai STATISTIK LATIH saja
    for pid, sub in df.groupby('partisipan'):
        tr = sub[sub['split'] == 'train']
        mu = tr[fitur].mean()
        sd = tr[fitur].std(ddof=0).replace(0, np.nan)
        df.loc[sub.index, fitur] = ((sub[fitur] - mu) / (sd + 1e-8)).fillna(0.0).values
    return df, fitur


# ============================================================
# 2) WINDOWING terpisah per split -> tensor
# ============================================================
def jendela(df, fitur, split):
    step = max(1, int(WINDOW * (1 - OVERLAP)))
    X, y, grp = [], [], []
    d = df[df['split'] == split]
    for _, sub in d.groupby(GRUP_KOND, dropna=False):
        arr = sub[fitur].to_numpy(np.float32); lab = sub['y'].to_numpy()
        pid = sub['partisipan'].iloc[0]
        for s in range(0, len(arr) - WINDOW + 1, step):
            X.append(arr[s:s+WINDOW])
            y.append(int(round(lab[s:s+WINDOW].mean())))
            grp.append(pid)
    return np.array(X), np.array(y), np.array(grp)


# ============================================================
# 3) MODEL (identik dgn skrip 04)
# ============================================================
def build_cnn(T,F):
    from tensorflow.keras import layers, models
    return models.Sequential([layers.Input((T,F)),
        layers.Conv1D(32,3,padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(64,3,padding='same',activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32,activation='relu'), layers.Dropout(0.3),
        layers.Dense(1,activation='sigmoid')], name="CNN")

def build_lstm(T,F):
    from tensorflow.keras import layers, models
    return models.Sequential([layers.Input((T,F)),
        layers.LSTM(64,return_sequences=True), layers.LSTM(32),
        layers.Dense(32,activation='relu'), layers.Dropout(0.3),
        layers.Dense(1,activation='sigmoid')], name="LSTM")

def build_cnnlstm(T,F):
    from tensorflow.keras import layers, models
    return models.Sequential([layers.Input((T,F)),
        layers.Conv1D(32,3,padding='same',activation='relu'),
        layers.BatchNormalization(), layers.LSTM(32),
        layers.Dense(32,activation='relu'), layers.Dropout(0.3),
        layers.Dense(1,activation='sigmoid')], name="CNN-LSTM")

MODELS = {"CNN":build_cnn, "LSTM":build_lstm, "CNN-LSTM":build_cnnlstm}


# ============================================================
# 4) MAIN
# ============================================================
def main():
    os.makedirs(DIR_RUN, exist_ok=True)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow.keras import optimizers
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 roc_auc_score, confusion_matrix)

    df, fitur = siapkan()
    Xtr,ytr,gtr = jendela(df, fitur, 'train')
    Xte,yte,gte = jendela(df, fitur, 'test')
    T,F = Xtr.shape[1], Xtr.shape[2]
    print(f"Latih: {Xtr.shape} | Uji: {Xte.shape} | fitur={fitur}")
    print(f"Subjek uji: {len(np.unique(gte))} | "
          f"kelas uji: atensi={int((yte==0).sum())}, stres={int((yte==1).sum())}")
    print(f"Paradigma: WITHIN-SUBJECT (kalibrasi) | epoch={EPOCHS} | quick={QUICK_TEST}\n")

    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=ytr)
    hist_simpan, oof, lat = {}, {}, {}
    per_subj_rows = []

    for nama, build in MODELS.items():
        print(f"--- Melatih {nama} ---")
        m = build(T,F)
        m.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])
        vb = 1 if (not QUICK_TEST) else 0
        t0=time.time()
        h = m.fit(Xtr,ytr, validation_data=(Xte,yte), epochs=EPOCHS,
                  batch_size=BATCH, class_weight={0:cw[0],1:cw[1]}, verbose=vb)
        tl=time.time()-t0
        t1=time.time(); prob=m.predict(Xte,verbose=0).ravel()
        ti=(time.time()-t1)/max(1,len(Xte))*1000
        yp=(prob>=0.5).astype(int)
        hist_simpan[nama]=h.history
        oof[nama]={'yt':yte,'yp':yp,'pr':prob}
        lat[nama]={'infer':ti,'param':int(m.count_params()),'latih':tl}
        # metrik per-subjek
        for pid in np.unique(gte):
            mask=gte==pid
            if mask.sum()<MIN_WIN_SUBJEK or len(np.unique(yte[mask]))<2:
                continue
            per_subj_rows.append({'partisipan':pid,'model':nama,
                'balanced_acc':balanced_accuracy_score(yte[mask],yp[mask]),
                'n_uji':int(mask.sum())})
        print(f"  balanced_acc(uji global)={balanced_accuracy_score(yte,yp):.4f}")

    subj_df=pd.DataFrame(per_subj_rows)
    subj_df.to_csv(os.path.join(DIR_RUN,"metrik_per_subjek.csv"),index=False)

    # ringkasan + permutasi
    rows=[]
    for nama in MODELS:
        yt=oof[nama]['yt']; yp=oof[nama]['yp']; pr=oof[nama]['pr']
        ba=balanced_accuracy_score(yt,yp)
        rng=np.random.default_rng(SEED)
        null=np.array([balanced_accuracy_score(rng.permutation(yt),yp)
                       for _ in range(N_PERMUTASI)])
        p=(1+np.sum(null>=ba))/(N_PERMUTASI+1)
        ps=subj_df[subj_df.model==nama]['balanced_acc']
        rows.append({'model':nama,
            'bacc_per_subjek_mean':round(ps.mean(),4),
            'bacc_per_subjek_std':round(ps.std(),4),
            'balanced_acc_global':round(ba,4),
            'f1_macro':round(f1_score(yt,yp,average='macro'),4),
            'auc':round(roc_auc_score(yt,pr),4),
            'perm_mean':round(null.mean(),4),'p_value':round(p,4),
            'latensi_ms':round(lat[nama]['infer'],3),
            'n_param':lat[nama]['param'],
            'waktu_latih_s':round(lat[nama]['latih'],2)})
    ring=pd.DataFrame(rows)
    ring.to_csv(os.path.join(DIR_RUN,"ringkasan_model.csv"),index=False)
    print("\n=== RINGKASAN MODEL (within-subject) ===")
    print(ring.to_string(index=False))

    _plot(subj_df, oof, hist_simpan, ring)
    with open(os.path.join(DIR_RUN,"INFO.txt"),"w",encoding="utf-8") as f:
        f.write(f"Run {STAMP} | WITHIN-SUBJECT | TRAIN_RATIO={TRAIN_RATIO}\n")
        f.write(f"Latih={Xtr.shape}, Uji={Xte.shape}, epoch={EPOCHS}, quick={QUICK_TEST}\n")
        f.write("Normalisasi dari statistik LATIH saja (anti-bocor).\n")
        f.write("Apple-to-apple: hanya arsitektur berbeda.\n")
    print("\n[SELESAI] Output di:", DIR_RUN)


# ============================================================
# 5) GRAFIK
# ============================================================
def _plot(subj_df, oof, hist, ring):
    from sklearn.metrics import confusion_matrix
    warna={'CNN':'#2c6fbb','LSTM':'#c0392b','CNN-LSTM':'#2e8b57'}

    # (a) kurva belajar: latih vs uji (memperlihatkan generalisasi per-epoch)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for nama,h in hist.items():
        ax[0].plot(h['loss'],color=warna[nama],label=f"{nama} latih")
        ax[0].plot(h['val_loss'],color=warna[nama],ls='--',label=f"{nama} uji")
        ax[1].plot(h['accuracy'],color=warna[nama],label=f"{nama} latih")
        ax[1].plot(h['val_accuracy'],color=warna[nama],ls='--',label=f"{nama} uji")
    ax[0].set_title("Loss (garis putus = uji)"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[1].set_title("Accuracy (garis putus = uji)"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("acc")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"01_kurva_belajar.png"),dpi=150); plt.close()

    # (b) boxplot balanced_acc PER-SUBJEK
    fig,ax=plt.subplots(figsize=(7,5))
    data=[subj_df[subj_df.model==m]['balanced_acc'] for m in warna]
    bp=ax.boxplot(data,labels=list(warna),patch_artist=True)
    for patch,m in zip(bp['boxes'],warna): patch.set_facecolor(warna[m]); patch.set_alpha(0.6)
    ax.axhline(0.5,color='k',ls=':',label='tebakan (0,5)')
    ax.set_ylabel("balanced accuracy per-subjek"); ax.legend()
    ax.set_title("Sebaran balanced accuracy antar-subjek (within-subject)")
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"02_boxplot_per_subjek.png"),dpi=150); plt.close()

    # (c) confusion matrix (uji global)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,nama in zip(axes,warna):
        cm=confusion_matrix(oof[nama]['yt'],oof[nama]['yp'])
        ax.imshow(cm,cmap='Blues')
        for (r,c),v in np.ndenumerate(cm):
            ax.text(c,r,int(v),ha='center',va='center',
                    color='white' if v>cm.max()/2 else 'black')
        ax.set_title(nama); ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['atensi','stres']); ax.set_yticklabels(['atensi','stres'])
        ax.set_xlabel('prediksi'); ax.set_ylabel('aktual')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"03_confusion_matrix.png"),dpi=150); plt.close()

    # (d) uji permutasi
    fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(len(ring))
    ax.bar(x,ring['balanced_acc_global'],color=[warna[m] for m in ring.model])
    ax.plot(x,ring['perm_mean'],'k_',ms=25,label='rata-rata acak')
    ax.axhline(0.5,color='grey',ls=':')
    for i,r in ring.iterrows(): ax.text(i,r['balanced_acc_global']+0.005,f"p={r['p_value']}",ha='center',fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ring.model); ax.set_ylabel("balanced accuracy (uji global)")
    ax.set_title("Uji permutasi (within-subject)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"04_uji_permutasi.png"),dpi=150); plt.close()

    # (e) PARETO benchmarking
    fig,ax=plt.subplots(figsize=(8,6))
    sizes=(ring['n_param'].values/ring['n_param'].max())*900+100
    for i,r in ring.iterrows():
        ax.scatter(r['latensi_ms'],r['balanced_acc_global'],s=sizes[i],
                   color=warna[r['model']],alpha=0.7,edgecolor='k',zorder=3)
        ax.annotate(f"{r['model']}\n({r['n_param']:,} param)",
                    (r['latensi_ms'],r['balanced_acc_global']),
                    xytext=(8,8),textcoords='offset points',fontsize=9)
    pts=ring.sort_values('latensi_ms'); frontier=[]; best=-1
    for _,r in pts.iterrows():
        if r['balanced_acc_global']>best: frontier.append((r['latensi_ms'],r['balanced_acc_global'])); best=r['balanced_acc_global']
    if len(frontier)>1:
        fx,fy=zip(*frontier); ax.plot(fx,fy,'--',color='grey',label='Pareto frontier')
    ax.axhline(0.5,color='grey',ls=':',alpha=0.6,label='tebakan (0,5)')
    ax.set_xlabel("latensi inferensi (ms/jendela) — makin kiri makin cepat")
    ax.set_ylabel("balanced accuracy — makin atas makin akurat")
    ax.set_title("Pareto Benchmarking Within-Subject (ukuran=parameter)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"05_pareto_benchmarking.png"),dpi=150); plt.close()

    # (f) peringkat
    fig,ax=plt.subplots(figsize=(7,5))
    r2=ring.sort_values('balanced_acc_global',ascending=False)
    ax.bar(r2['model'],r2['balanced_acc_global'],yerr=r2['bacc_per_subjek_std'],
           color=[warna[m] for m in r2.model],capsize=5)
    ax.axhline(0.5,color='k',ls=':'); ax.set_ylabel("balanced accuracy (uji global)")
    ax.set_title("Peringkat Model (within-subject)")
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"06_peringkat_model.png"),dpi=150); plt.close()


if __name__ == "__main__":
    main()
