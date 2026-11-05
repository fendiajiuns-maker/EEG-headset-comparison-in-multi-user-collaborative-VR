# -*- coding: utf-8 -*-
r"""
04c_klasifikasi_task_baseline.py
============================================================
KONTROL POSITIF: klasifikasi kontras yang perangkatnya TERBUKTI mampu.
Pelengkap skrip 04 (atensi/stres, cross-subject) & 04b (within-subject).

Latar: uji kelayakan menunjukkan reaktivitas alfa (mata tertutup > terbuka)
signifikan pada ~97% peserta (r=0,85). Jika pipeline yang SAMA berhasil
mengklasifikasi kontras ini, itu membuktikan kode/metode benar, dan kegagalan
atensi/stres murni karena sinyalnya tak ada — bukan pipeline yang rusak.

TARGET (pilih di PENGATURAN):
  "MATA"          : MataTertutup (1) vs MataTerbuka (0)  <- kontrol positif utama
  "TASK_BASELINE" : task Belajar/Quiz (1) vs baseline mata (0)

Evaluasi: cross-subject (GroupKFold / LOSO) — sama ketatnya dgn skrip 04.
Apple-to-apple: window/normalisasi/split/epoch/batch/lr/class_weight identik;
hanya arsitektur berbeda.

Cara pakai:
  1. pip install tensorflow scikit-learn pandas numpy matplotlib
  2. Sesuaikan PENGATURAN (terutama TARGET).
  3. python 04c_klasifikasi_task_baseline.py
============================================================
"""
import os, time, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#  PENGATURAN
# ============================================================
BASE_DIR   = r"D:\python vscode\venv\12_klasifikasieeg-2026"
FILE_DATA  = os.path.join(BASE_DIR, "data_bersih", "MASTER_EEG_bersih.csv")
DIR_HASIL  = os.path.join(BASE_DIR, "hasil_klasifikasi_kontrol_61baru")

TARGET     = "MATA"        # "MATA" (kontrol positif) atau "TASK_BASELINE"
QUICK_TEST = False          # True: KFOLD 5 + 20 epoch. False: LOSO + 100 epoch.

WINDOW=10; OVERLAP=0.5; PAKAI_ESENSE=True; LOG_BAND=True
EPOCHS=100; BATCH=32; LEARNING_RATE=1e-3; SEED=42
CV_MODE="LOSO"; N_SPLITS=5; N_PERMUTASI=1000
# ============================================================

if QUICK_TEST:
    EPOCHS=20; CV_MODE="KFOLD"
STAMP=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RUN=os.path.join(DIR_HASIL, f"run_{TARGET}_{STAMP}")
SIG_BAND=['Delta','Theta','Low Alpha','High Alpha','Low Beta','High Beta','Low Gamma','Mid Gamma']
ESENSE=['Attention','Meditation']
GRUP_KOND=['partisipan','versi_vr','kolaborasi','aktivitas']
np.random.seed(SEED)


# ============================================================
# 1) SIAPKAN DATA sesuai TARGET
# ============================================================
def siapkan():
    df=pd.read_csv(FILE_DATA)
    # buat label_emosi bila belum ada (file proyek 12 hanya punya 'aktivitas')
    if 'label_emosi' not in df.columns:
        df['label_emosi'] = df['aktivitas'].astype(str).map(
            {'Belajar':'atensi','Quiz':'stres',
             'MataTerbuka':'baseline','MataTertutup':'baseline'}).fillna('lain')
    if TARGET=="MATA":
        df=df[df['aktivitas'].isin(['MataTerbuka','MataTertutup'])].copy()
        df['y']=(df['aktivitas']=='MataTertutup').astype(int)   # tertutup=1
        kelas=['MataTerbuka','MataTertutup']
    elif TARGET=="TASK_BASELINE":
        df=df[df['label_emosi'].isin(['atensi','stres','baseline'])].copy()
        df['y']=(df['label_emosi']!='baseline').astype(int)      # task=1
        kelas=['baseline','task']
    else:
        raise ValueError("TARGET harus 'MATA' atau 'TASK_BASELINE'")

    fitur=SIG_BAND+(ESENSE if PAKAI_ESENSE else [])
    for c in fitur: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=fitur).reset_index(drop=True)
    df[fitur]=df[fitur].astype('float64')
    if LOG_BAND:
        for c in SIG_BAND: df[c]=np.log1p(df[c].clip(lower=0))
    # normalisasi z-score per-peserta
    gmean=df.groupby('partisipan')[fitur].transform('mean')
    gstd =df.groupby('partisipan')[fitur].transform('std').replace(0,np.nan)
    df[fitur]=((df[fitur]-gmean)/(gstd+1e-8)).fillna(0.0)
    return df, fitur, kelas


def jendela(df, fitur):
    step=max(1,int(WINDOW*(1-OVERLAP)))
    X,y,grp=[],[],[]
    for _,sub in df.groupby(GRUP_KOND,dropna=False):
        arr=sub[fitur].to_numpy(np.float32); lab=sub['y'].to_numpy()
        pid=sub['partisipan'].iloc[0]
        for s in range(0,len(arr)-WINDOW+1,step):
            X.append(arr[s:s+WINDOW]); y.append(int(round(lab[s:s+WINDOW].mean()))); grp.append(pid)
    return np.array(X),np.array(y),np.array(grp)


# ============================================================
# 2) MODEL (identik dgn skrip 04)
# ============================================================
def build_cnn(T,F):
    from tensorflow.keras import layers, models
    return models.Sequential([layers.Input((T,F)),
        layers.Conv1D(32,3,padding='same',activation='relu'), layers.BatchNormalization(),
        layers.Conv1D(64,3,padding='same',activation='relu'), layers.GlobalAveragePooling1D(),
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
        layers.Conv1D(32,3,padding='same',activation='relu'), layers.BatchNormalization(),
        layers.LSTM(32), layers.Dense(32,activation='relu'), layers.Dropout(0.3),
        layers.Dense(1,activation='sigmoid')], name="CNN-LSTM")
MODELS={"CNN":build_cnn,"LSTM":build_lstm,"CNN-LSTM":build_cnnlstm}


# ============================================================
# 3) MAIN
# ============================================================
def main():
    os.makedirs(DIR_RUN,exist_ok=True)
    import tensorflow as tf; tf.random.set_seed(SEED)
    from tensorflow.keras import optimizers
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.metrics import (balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix)

    df,fitur,kelas=siapkan()
    X,y,grp=jendela(df,fitur)
    T,F=X.shape[1],X.shape[2]; n_subj=len(np.unique(grp))
    print(f"TARGET={TARGET} | kelas={kelas}")
    print(f"Jendela: {X.shape} | subjek={n_subj} | "
          f"kelas0={int((y==0).sum())}, kelas1={int((y==1).sum())}")

    if CV_MODE=="LOSO": splitter=LeaveOneGroupOut(); n_fold=n_subj
    else: n_fold=min(N_SPLITS,n_subj); splitter=GroupKFold(n_splits=n_fold)
    print(f"CV: {CV_MODE} ({n_fold} fold) | epoch={EPOCHS} | quick={QUICK_TEST}\n")

    baris=[]; oof={k:{'yt':[],'yp':[],'pr':[]} for k in MODELS}
    hist_simpan={k:None for k in MODELS}; lat={k:{'infer':[],'param':0} for k in MODELS}

    for i,(tr,te) in enumerate(splitter.split(X,y,grp),1):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        if len(np.unique(ytr))<2 or len(np.unique(yte))<2: continue
        cw=compute_class_weight('balanced',classes=np.array([0,1]),y=ytr)
        for nama,build in MODELS.items():
            vb=1 if (i==1 and not QUICK_TEST) else 0
            if i==1: print(f"--- Fold {i} · {nama} ---")
            m=build(T,F); m.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                                    loss='binary_crossentropy',metrics=['accuracy'])
            h=m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=BATCH,
                    class_weight={0:cw[0],1:cw[1]},verbose=vb)
            t1=time.time(); prob=m.predict(Xte,verbose=0).ravel()
            ti=(time.time()-t1)/max(1,len(Xte))*1000
            yp=(prob>=0.5).astype(int)
            baris.append({'fold':i,'model':nama,
                'balanced_acc':balanced_accuracy_score(yte,yp),
                'f1_macro':f1_score(yte,yp,average='macro'),
                'auc':roc_auc_score(yte,prob) if len(np.unique(yte))>1 else np.nan})
            oof[nama]['yt']+=list(yte); oof[nama]['yp']+=list(yp); oof[nama]['pr']+=list(prob)
            lat[nama]['infer'].append(ti); lat[nama]['param']=int(m.count_params())
            if hist_simpan[nama] is None: hist_simpan[nama]=h.history
        print(f"[fold {i}/{n_fold}] selesai")

    fold_df=pd.DataFrame(baris); fold_df.to_csv(os.path.join(DIR_RUN,"metrik_per_fold.csv"),index=False)
    ring=(fold_df.groupby('model')[['balanced_acc','f1_macro','auc']].agg(['mean','std']).round(4))
    ring.columns=['_'.join(c) for c in ring.columns]; ring=ring.reset_index()
    rows=[]
    for nama in MODELS:
        yt=np.array(oof[nama]['yt']); yp=np.array(oof[nama]['yp'])
        ba=balanced_accuracy_score(yt,yp)
        rng=np.random.default_rng(SEED)
        null=np.array([balanced_accuracy_score(rng.permutation(yt),yp) for _ in range(N_PERMUTASI)])
        p=(1+np.sum(null>=ba))/(N_PERMUTASI+1)
        rows.append({'model':nama,'balanced_acc_oof':round(ba,4),'perm_mean':round(null.mean(),4),
                     'p_value':round(p,4),'latensi_ms':round(np.mean(lat[nama]['infer']),3),
                     'n_param':lat[nama]['param']})
    ringkas=ring.merge(pd.DataFrame(rows),on='model')
    ringkas.to_csv(os.path.join(DIR_RUN,"ringkasan_model.csv"),index=False)
    print("\n=== RINGKASAN MODEL ===")
    print(ringkas.to_string(index=False))

    _plot(fold_df,oof,hist_simpan,ringkas,kelas)
    with open(os.path.join(DIR_RUN,"INFO.txt"),"w",encoding="utf-8") as f:
        f.write(f"Run {STAMP} | TARGET={TARGET} ({kelas}) | CV={CV_MODE} | epoch={EPOCHS}\n")
        f.write(f"Jendela={X.shape}. Apple-to-apple: hanya arsitektur berbeda.\n")
    print("\n[SELESAI] Output di:", DIR_RUN)


# ============================================================
# 4) GRAFIK
# ============================================================
def _plot(fold_df,oof,hist,ringkas,kelas):
    from sklearn.metrics import confusion_matrix
    warna={'CNN':'#2c6fbb','LSTM':'#c0392b','CNN-LSTM':'#2e8b57'}
    # kurva belajar
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for nama,h in hist.items():
        if h is None: continue
        ax[0].plot(h['loss'],color=warna[nama],label=nama); ax[1].plot(h['accuracy'],color=warna[nama],label=nama)
    ax[0].set_title("Loss (fold-1)"); ax[0].set_xlabel("epoch"); ax[1].set_title("Accuracy (fold-1)"); ax[1].set_xlabel("epoch")
    ax[0].legend(); ax[1].legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"01_kurva_belajar.png"),dpi=150); plt.close()
    # boxplot
    fig,ax=plt.subplots(figsize=(7,5))
    data=[fold_df[fold_df.model==m]['balanced_acc'] for m in warna]
    bp=ax.boxplot(data,labels=list(warna),patch_artist=True)
    for patch,m in zip(bp['boxes'],warna): patch.set_facecolor(warna[m]); patch.set_alpha(0.6)
    ax.axhline(0.5,color='k',ls=':',label='tebakan (0,5)'); ax.set_ylabel("balanced accuracy")
    ax.set_title(f"Sebaran balanced accuracy antar-fold ({kelas[1]} vs {kelas[0]})"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"02_boxplot_balanced_acc.png"),dpi=150); plt.close()
    # confusion
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,nama in zip(axes,warna):
        cm=confusion_matrix(oof[nama]['yt'],oof[nama]['yp'])
        ax.imshow(cm,cmap='Blues')
        for (r,c),v in np.ndenumerate(cm): ax.text(c,r,int(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
        ax.set_title(nama); ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(kelas); ax.set_yticklabels(kelas); ax.set_xlabel('prediksi'); ax.set_ylabel('aktual')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"03_confusion_matrix.png"),dpi=150); plt.close()
    # permutasi
    fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(len(ringkas))
    ax.bar(x,ringkas['balanced_acc_oof'],color=[warna[m] for m in ringkas.model])
    ax.plot(x,ringkas['perm_mean'],'k_',ms=25,label='rata-rata acak'); ax.axhline(0.5,color='grey',ls=':')
    for i,r in ringkas.iterrows(): ax.text(i,r['balanced_acc_oof']+0.01,f"p={r['p_value']}",ha='center',fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ringkas.model); ax.set_ylabel("balanced accuracy"); ax.set_ylim(0,1)
    ax.set_title("Uji permutasi (OOF)"); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"04_uji_permutasi.png"),dpi=150); plt.close()
    # pareto
    fig,ax=plt.subplots(figsize=(8,6))
    sizes=(ringkas['n_param'].values/ringkas['n_param'].max())*900+100
    for i,r in ringkas.iterrows():
        ax.scatter(r['latensi_ms'],r['balanced_acc_oof'],s=sizes[i],color=warna[r['model']],alpha=0.7,edgecolor='k',zorder=3)
        ax.annotate(f"{r['model']}\n({r['n_param']:,} param)",(r['latensi_ms'],r['balanced_acc_oof']),xytext=(8,8),textcoords='offset points',fontsize=9)
    pts=ringkas.sort_values('latensi_ms'); frontier=[]; best=-1
    for _,r in pts.iterrows():
        if r['balanced_acc_oof']>best: frontier.append((r['latensi_ms'],r['balanced_acc_oof'])); best=r['balanced_acc_oof']
    if len(frontier)>1: fx,fy=zip(*frontier); ax.plot(fx,fy,'--',color='grey',label='Pareto frontier')
    ax.axhline(0.5,color='grey',ls=':',alpha=0.6,label='tebakan (0,5)')
    ax.set_xlabel("latensi inferensi (ms/jendela) — makin kiri makin cepat")
    ax.set_ylabel("balanced accuracy — makin atas makin akurat")
    ax.set_title(f"Pareto Benchmarking ({kelas[1]} vs {kelas[0]})"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"05_pareto_benchmarking.png"),dpi=150); plt.close()
    # peringkat
    fig,ax=plt.subplots(figsize=(7,5)); r2=ringkas.sort_values('balanced_acc_oof',ascending=False)
    ax.bar(r2['model'],r2['balanced_acc_oof'],yerr=r2['balanced_acc_std'],color=[warna[m] for m in r2.model],capsize=5)
    ax.axhline(0.5,color='k',ls=':'); ax.set_ylabel("balanced accuracy (OOF)"); ax.set_ylim(0,1)
    ax.set_title("Peringkat Model"); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"06_peringkat_model.png"),dpi=150); plt.close()


if __name__ == "__main__":
    main()
