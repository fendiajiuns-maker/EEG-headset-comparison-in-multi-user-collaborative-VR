# -*- coding: utf-8 -*-
r"""
04b_muse_klasifikasi.py  —  WITHIN-SUBJECT (kalibrasi per-pengguna)
============================================================
Klasifikasi ATENSI vs STRES dari EEG Muse 2, paradigma within-subject:
tiap peserta dibagi TEMPORAL (70% awal latih, 30% akhir uji); normalisasi
dihitung HANYA dari bagian latih (anti-bocor). Proses per-model & per-epoch
terlihat. Setara skrip 04b Shicyrai, disesuaikan untuk Muse.

SKENARIO fitur (toggle): "SEMUA" (20 pita + FAA) atau "TANPA_FRONTAL" (TP9/TP10).
Metrik: balanced accuracy, F1-macro, AKURASI, AUC + uji permutasi.

Cara pakai:
  pip install tensorflow scikit-learn pandas numpy matplotlib
  Atur SKENARIO & QUICK_TEST, lalu: python 04b_muse_klasifikasi.py
============================================================
"""
import os, time, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR  = r"D:\python vscode\venv\14_klasifikasieeg_muse_2026"
FILE_DATA = os.path.join(BASE_DIR, "data_bersih_muse", "MASTER_MUSE_bersih.csv")
DIR_HASIL = os.path.join(BASE_DIR, "hasil_klasifikasi_muse_within")

SKENARIO="SEMUA"; QUICK_TEST=False
WIN_SAMPLES=20; STEP=10; TRAIN_RATIO=0.7
EPOCHS=100; BATCH=32; LR=1e-3; SEED=42; N_PERMUTASI=1000; MIN_WIN_SUBJEK=5
if QUICK_TEST: EPOCHS=20
STAMP=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RUN=os.path.join(DIR_HASIL,f"run_{SKENARIO}_{STAMP}")
KANAL=['TP9','AF7','AF8','TP10']; PITA=['Delta','Theta','Alpha','Beta','Gamma']
BAND=[f"{k}_{p}" for k in KANAL for p in PITA]
GRUP=['partisipan','versi_vr','kolaborasi','aktivitas']
np.random.seed(SEED)


def siapkan():
    df0=pd.read_csv(FILE_DATA)
    df=df0[df0['label_emosi'].isin(['atensi','stres'])].copy()
    df['y']=(df['label_emosi']=='stres').astype(int)
    for c in BAND: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=BAND).reset_index(drop=True); df[BAND]=df[BAND].astype('float64')
    for c in BAND: df[c]=np.log1p(df[c].clip(lower=0))
    if SKENARIO=="SEMUA":
        fitur=list(BAND)
        df['FAA_alpha']=np.log(df['AF8_Alpha'].clip(lower=1e-6))-np.log(df['AF7_Alpha'].clip(lower=1e-6))
        df['FAA_theta']=np.log(df['AF8_Theta'].clip(lower=1e-6))-np.log(df['AF7_Theta'].clip(lower=1e-6))
        fitur+=['FAA_alpha','FAA_theta']
    else:
        fitur=[f"{k}_{p}" for k in ['TP9','TP10'] for p in PITA]
    # split temporal per blok
    df['split']='train'
    for _,idx in df.groupby(GRUP,dropna=False).groups.items():
        idx=list(idx); ntr=int(np.ceil(len(idx)*TRAIN_RATIO))
        for i in idx[ntr:]: df.at[i,'split']='test'
    # normalisasi z-score per-peserta dari statistik LATIH
    for pid,sub in df.groupby('partisipan'):
        tr=sub[sub['split']=='train']; mu=tr[fitur].mean(); sd=tr[fitur].std(ddof=0).replace(0,np.nan)
        df.loc[sub.index,fitur]=((sub[fitur]-mu)/(sd+1e-8)).fillna(0.0).values
    return df,fitur

def jendela(df,fitur,split):
    X,y,grp=[],[],[]
    for keys,sub in df[df['split']==split].groupby(GRUP,dropna=False):
        arr=sub[fitur].to_numpy(np.float32); lab=sub['y'].to_numpy(); pid=sub['partisipan'].iloc[0]
        for s in range(0,len(arr)-WIN_SAMPLES+1,STEP):
            X.append(arr[s:s+WIN_SAMPLES]); y.append(int(round(lab[s:s+WIN_SAMPLES].mean()))); grp.append(pid)
    return np.array(X),np.array(y),np.array(grp)

def build_cnn(T,F):
    from tensorflow.keras import layers,models
    return models.Sequential([layers.Input((T,F)),layers.Conv1D(32,3,padding='same',activation='relu'),
        layers.BatchNormalization(),layers.Conv1D(64,3,padding='same',activation='relu'),
        layers.GlobalAveragePooling1D(),layers.Dense(32,activation='relu'),layers.Dropout(0.3),
        layers.Dense(1,activation='sigmoid')],name="CNN")
def build_lstm(T,F):
    from tensorflow.keras import layers,models
    return models.Sequential([layers.Input((T,F)),layers.LSTM(64,return_sequences=True),layers.LSTM(32),
        layers.Dense(32,activation='relu'),layers.Dropout(0.3),layers.Dense(1,activation='sigmoid')],name="LSTM")
def build_cnnlstm(T,F):
    from tensorflow.keras import layers,models
    return models.Sequential([layers.Input((T,F)),layers.Conv1D(32,3,padding='same',activation='relu'),
        layers.BatchNormalization(),layers.LSTM(32),layers.Dense(32,activation='relu'),
        layers.Dropout(0.3),layers.Dense(1,activation='sigmoid')],name="CNN-LSTM")
MODELS={"CNN":build_cnn,"LSTM":build_lstm,"CNN-LSTM":build_cnnlstm}


def main():
    os.makedirs(DIR_RUN,exist_ok=True)
    import tensorflow as tf; tf.random.set_seed(SEED)
    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g,True)
        except Exception: pass
    from tensorflow.keras import optimizers
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import balanced_accuracy_score,f1_score,accuracy_score,roc_auc_score,confusion_matrix

    df,fitur=siapkan()
    Xtr,ytr,_=jendela(df,fitur,'train'); Xte,yte,gte=jendela(df,fitur,'test')
    T,F=Xtr.shape[1],Xtr.shape[2]
    print(f"SKENARIO={SKENARIO} ({F} fitur) | Latih {Xtr.shape} | Uji {Xte.shape} | subjek uji {len(np.unique(gte))}")
    print(f"WITHIN-SUBJECT | epoch={EPOCHS} | quick={QUICK_TEST}\n")
    cw=compute_class_weight('balanced',classes=np.array([0,1]),y=ytr)

    oof={}; hist={}; lat={}; persubj=[]
    for nm,bd in MODELS.items():
        print(f"--- Melatih {nm} ---")
        m=bd(T,F); m.compile(optimizer=optimizers.Adam(LR),loss='binary_crossentropy',metrics=['accuracy'])
        vb=1 if not QUICK_TEST else 0
        t0=time.time(); h=m.fit(Xtr,ytr,validation_data=(Xte,yte),epochs=EPOCHS,batch_size=BATCH,
                                class_weight={0:cw[0],1:cw[1]},verbose=vb); tl=time.time()-t0
        t1=time.time(); pr=m.predict(Xte,verbose=0).ravel(); ti=(time.time()-t1)/max(1,len(Xte))*1000
        yp=(pr>=0.5).astype(int)
        oof[nm]={'yt':yte,'yp':yp,'pr':pr}; hist[nm]=h.history; lat[nm]={'ti':ti,'np':int(m.count_params())}
        for pid in np.unique(gte):
            mk=gte==pid
            if mk.sum()>=MIN_WIN_SUBJEK and len(np.unique(yte[mk]))>1:
                persubj.append({'partisipan':pid,'model':nm,'balanced_acc':balanced_accuracy_score(yte[mk],yp[mk])})
        print(f"  bAcc={balanced_accuracy_score(yte,yp):.4f} F1={f1_score(yte,yp,average='macro'):.4f} "
              f"Acc={accuracy_score(yte,yp):.4f} AUC={roc_auc_score(yte,pr):.4f}")

    subj_df=pd.DataFrame(persubj); subj_df.to_csv(os.path.join(DIR_RUN,"metrik_per_subjek.csv"),index=False)
    rows=[]
    for nm in MODELS:
        yt=oof[nm]['yt']; yp=oof[nm]['yp']; pr=oof[nm]['pr']; ba=balanced_accuracy_score(yt,yp)
        rng=np.random.default_rng(SEED)
        null=np.array([balanced_accuracy_score(rng.permutation(yt),yp) for _ in range(N_PERMUTASI)])
        ps=subj_df[subj_df.model==nm]['balanced_acc']
        rows.append({'model':nm,'balanced_acc':round(ba,4),'f1_macro':round(f1_score(yt,yp,average='macro'),4),
                     'akurasi':round(accuracy_score(yt,yp),4),'auc':round(roc_auc_score(yt,pr),4),
                     'bacc_per_subjek_std':round(ps.std(),4) if len(ps) else np.nan,
                     'p_value':round((1+np.sum(null>=ba))/(N_PERMUTASI+1),4),
                     'latensi_ms':round(lat[nm]['ti'],3),'n_param':lat[nm]['np']})
    ringkas=pd.DataFrame(rows); ringkas.to_csv(os.path.join(DIR_RUN,"ringkasan_model.csv"),index=False)
    print("\n=== RINGKASAN MODEL (within-subject) ===\n"+ringkas.to_string(index=False))
    _plot(subj_df,oof,hist,ringkas)
    with open(os.path.join(DIR_RUN,"INFO.txt"),"w",encoding="utf-8") as f:
        f.write(f"Muse WITHIN-SUBJECT | SKENARIO={SKENARIO} | epoch={EPOCHS} | TRAIN_RATIO={TRAIN_RATIO}\n")
        f.write("Normalisasi dari statistik latih saja (anti-bocor).\n")
    print("\n[SELESAI] Output di:",DIR_RUN)


def _plot(subj_df,oof,hist,ringkas):
    from sklearn.metrics import confusion_matrix
    W={'CNN':'#2c6fbb','LSTM':'#c0392b','CNN-LSTM':'#2e8b57'}
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for nm,h in hist.items():
        ax[0].plot(h['loss'],color=W[nm],label=f"{nm} latih"); ax[0].plot(h['val_loss'],color=W[nm],ls='--',label=f"{nm} uji")
        ax[1].plot(h['accuracy'],color=W[nm],label=f"{nm} latih"); ax[1].plot(h['val_accuracy'],color=W[nm],ls='--',label=f"{nm} uji")
    ax[0].set_title("Loss (putus=uji)"); ax[1].set_title("Accuracy (putus=uji)")
    for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"01_kurva_belajar.png"),dpi=150); plt.close()

    if not subj_df.empty:
        fig,ax=plt.subplots(figsize=(7,5))
        bp=ax.boxplot([subj_df[subj_df.model==m]['balanced_acc'] for m in W],labels=list(W),patch_artist=True)
        for p,m in zip(bp['boxes'],W): p.set_facecolor(W[m]); p.set_alpha(0.6)
        ax.axhline(0.5,color='k',ls=':',label='tebakan'); ax.set_ylabel("balanced accuracy per-subjek"); ax.legend()
        ax.set_title(f"Sebaran antar-subjek ({SKENARIO})"); plt.tight_layout()
        plt.savefig(os.path.join(DIR_RUN,"02_boxplot_per_subjek.png"),dpi=150); plt.close()

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for a,nm in zip(axes,W):
        cm=confusion_matrix(oof[nm]['yt'],oof[nm]['yp']); a.imshow(cm,cmap='Blues')
        for (r,c),v in np.ndenumerate(cm): a.text(c,r,int(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
        a.set_title(nm); a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xticklabels(['atensi','stres']); a.set_yticklabels(['atensi','stres'])
        a.set_xlabel('prediksi'); a.set_ylabel('aktual')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"03_confusion_matrix.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(len(ringkas))
    ax.bar(x,ringkas['balanced_acc'],color=[W[m] for m in ringkas.model]); ax.axhline(0.5,color='grey',ls=':')
    for i,r in ringkas.iterrows(): ax.text(i,r['balanced_acc']+0.005,f"p={r['p_value']}",ha='center',fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ringkas.model); ax.set_ylabel("balanced accuracy"); ax.set_ylim(0,1)
    ax.set_title("Uji permutasi"); plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"04_uji_permutasi.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(8,6)); sizes=(ringkas['n_param']/ringkas['n_param'].max())*900+100
    for i,r in ringkas.iterrows():
        ax.scatter(r['latensi_ms'],r['balanced_acc'],s=sizes[i],color=W[r['model']],alpha=0.75,edgecolor='k',zorder=3)
        ax.annotate(f"{r['model']}\n({r['n_param']:,} p)",(r['latensi_ms'],r['balanced_acc']),xytext=(8,8),textcoords='offset points',fontsize=9)
    ax.axhline(0.5,color='grey',ls=':',label='tebakan'); ax.set_xlabel("latensi (ms/jendela)"); ax.set_ylabel("balanced accuracy")
    ax.set_title(f"Pareto Benchmarking ({SKENARIO})"); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"05_pareto_benchmarking.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(7,5)); r2=ringkas.sort_values('balanced_acc',ascending=False)
    ax.bar(r2['model'],r2['balanced_acc'],yerr=r2['bacc_per_subjek_std'],color=[W[m] for m in r2.model],capsize=5)
    ax.axhline(0.5,color='k',ls=':'); ax.set_ylabel("balanced accuracy"); ax.set_ylim(0,1); ax.set_title("Peringkat Model")
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"06_peringkat_model.png"),dpi=150); plt.close()


if __name__=="__main__":
    main()
