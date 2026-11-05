# -*- coding: utf-8 -*-
r"""
04c_muse_klasifikasi.py  —  KONTROL POSITIF (mata tertutup vs terbuka)
============================================================
Klasifikasi kontras fisiologis mata TERTUTUP vs TERBUKA (efek Berger) dari
EEG Muse 2, cross-subject. Kontrol positif: jika berhasil, membuktikan
pipeline benar. Setara skrip 04c Shicyrai, disesuaikan untuk Muse.

SKENARIO fitur (toggle): "SEMUA" (20 pita + FAA) atau "TANPA_FRONTAL" (TP9/TP10).
Metrik: balanced accuracy, F1-macro, AKURASI, AUC + uji permutasi.

Cara pakai:
  pip install tensorflow scikit-learn pandas numpy matplotlib
  Atur SKENARIO & QUICK_TEST, lalu: python 04c_muse_klasifikasi.py
============================================================
"""
import os, time, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR  = r"D:\python vscode\venv\14_klasifikasieeg_muse_2026"
FILE_DATA = os.path.join(BASE_DIR, "data_bersih_muse", "MASTER_MUSE_bersih.csv")
DIR_HASIL = os.path.join(BASE_DIR, "hasil_klasifikasi_muse_kontrol")

SKENARIO="SEMUA"; QUICK_TEST=False
WIN_SAMPLES=20; STEP=10
EPOCHS=100; BATCH=32; LR=1e-3; EARLY_STOP=False; SEED=42
CV_MODE="LOSO"; N_SPLITS=5; N_PERMUTASI=1000
KELAS=['MataTerbuka','MataTertutup']    # 0, 1
if QUICK_TEST: EPOCHS=20; CV_MODE="KFOLD"
STAMP=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RUN=os.path.join(DIR_HASIL,f"run_{SKENARIO}_{STAMP}")
KANAL=['TP9','AF7','AF8','TP10']; PITA=['Delta','Theta','Alpha','Beta','Gamma']
BAND=[f"{k}_{p}" for k in KANAL for p in PITA]
GRUP=['partisipan','versi_vr','kolaborasi','aktivitas']
np.random.seed(SEED)


def siapkan():
    df0=pd.read_csv(FILE_DATA)
    df=df0[df0['aktivitas'].isin(KELAS)].copy()
    df['y']=(df['aktivitas']=='MataTertutup').astype(int)
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
    gm=df.groupby('partisipan')[fitur].transform('mean')
    gs=df.groupby('partisipan')[fitur].transform('std').replace(0,np.nan)
    df[fitur]=((df[fitur]-gm)/(gs+1e-8)).fillna(0.0)
    return df,fitur

def jendela(df,fitur):
    X,y,grp=[],[],[]
    for keys,sub in df.groupby(GRUP,dropna=False):
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

def latih(build,Xtr,ytr,Xte,T,F,verbose):
    from tensorflow.keras import optimizers
    from sklearn.utils.class_weight import compute_class_weight
    m=build(T,F); m.compile(optimizer=optimizers.Adam(LR),loss='binary_crossentropy',metrics=['accuracy'])
    cw=compute_class_weight('balanced',classes=np.array([0,1]),y=ytr)
    t0=time.time(); h=m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=BATCH,class_weight={0:cw[0],1:cw[1]},verbose=verbose); tl=time.time()-t0
    t1=time.time(); pr=m.predict(Xte,verbose=0).ravel(); ti=(time.time()-t1)/max(1,len(Xte))*1000
    return pr,h.history,tl,ti,int(m.count_params())


def main():
    os.makedirs(DIR_RUN,exist_ok=True)
    import tensorflow as tf; tf.random.set_seed(SEED)
    for g in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(g,True)
        except Exception: pass
    from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score,f1_score,accuracy_score,roc_auc_score,confusion_matrix

    df,fitur=siapkan(); X,y,grp=jendela(df,fitur); T,F=X.shape[1],X.shape[2]; ns=len(np.unique(grp))
    print(f"KONTROL mata | SKENARIO={SKENARIO} ({F} fitur) | Jendela {X.shape} | subjek {ns}")
    print(f"kelas: {KELAS[0]}={int((y==0).sum())}, {KELAS[1]}={int((y==1).sum())}")
    spl=LeaveOneGroupOut() if CV_MODE=="LOSO" else GroupKFold(min(N_SPLITS,ns)); nfold=ns if CV_MODE=="LOSO" else min(N_SPLITS,ns)
    print(f"CV={CV_MODE} ({nfold} fold) | epoch={EPOCHS} | quick={QUICK_TEST}\n")

    baris=[]; oof={k:{'yt':[],'yp':[],'pr':[]} for k in MODELS}; hist={k:None for k in MODELS}; lat={k:{'ti':[],'np':0} for k in MODELS}
    for i,(tr,te) in enumerate(spl.split(X,y,grp),1):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        for nm,bd in MODELS.items():
            vb=1 if (i==1 and not QUICK_TEST) else 0
            if i==1: print(f"--- Fold {i} · {nm} ---")
            pr,h,tl,ti,npar=latih(bd,X[tr],y[tr],X[te],T,F,vb); yp=(pr>=0.5).astype(int)
            baris.append({'fold':i,'model':nm,'balanced_acc':balanced_accuracy_score(y[te],yp),
                'f1_macro':f1_score(y[te],yp,average='macro'),'akurasi':accuracy_score(y[te],yp),
                'auc':roc_auc_score(y[te],pr) if len(np.unique(y[te]))>1 else np.nan})
            oof[nm]['yt']+=list(y[te]); oof[nm]['yp']+=list(yp); oof[nm]['pr']+=list(pr)
            lat[nm]['ti'].append(ti); lat[nm]['np']=npar
            if hist[nm] is None: hist[nm]=h
        print(f"[fold {i}/{nfold}] selesai")

    fold_df=pd.DataFrame(baris); fold_df.to_csv(os.path.join(DIR_RUN,"metrik_per_fold.csv"),index=False)
    ring=fold_df.groupby('model')[['balanced_acc','f1_macro','akurasi','auc']].agg(['mean','std']).round(4)
    ring.columns=['_'.join(c) for c in ring.columns]; ring=ring.reset_index()
    rows=[]
    for nm in MODELS:
        yt=np.array(oof[nm]['yt']); yp=np.array(oof[nm]['yp']); ba=balanced_accuracy_score(yt,yp)
        rng=np.random.default_rng(SEED)
        null=np.array([balanced_accuracy_score(rng.permutation(yt),yp) for _ in range(N_PERMUTASI)])
        rows.append({'model':nm,'balanced_acc_oof':round(ba,4),'p_value':round((1+np.sum(null>=ba))/(N_PERMUTASI+1),4),
                     'latensi_ms':round(np.mean(lat[nm]['ti']),3),'n_param':lat[nm]['np']})
    ringkas=ring.merge(pd.DataFrame(rows),on='model'); ringkas.to_csv(os.path.join(DIR_RUN,"ringkasan_model.csv"),index=False)
    print("\n=== RINGKASAN MODEL ===\n"+ringkas.to_string(index=False))
    _plot(fold_df,oof,hist,ringkas)
    with open(os.path.join(DIR_RUN,"INFO.txt"),"w",encoding="utf-8") as f:
        f.write(f"Muse KONTROL mata | SKENARIO={SKENARIO} | CV={CV_MODE} | epoch={EPOCHS} | Jendela={X.shape}\n")
    print("\n[SELESAI] Output di:",DIR_RUN)


def _plot(fold_df,oof,hist,ringkas):
    from sklearn.metrics import confusion_matrix
    W={'CNN':'#2c6fbb','LSTM':'#c0392b','CNN-LSTM':'#2e8b57'}
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for nm,h in hist.items():
        if h: ax[0].plot(h['loss'],color=W[nm],label=nm); ax[1].plot(h['accuracy'],color=W[nm],label=nm)
    ax[0].set_title("Loss (fold-1)"); ax[1].set_title("Accuracy (fold-1)")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"01_kurva_belajar.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(7,5))
    bp=ax.boxplot([fold_df[fold_df.model==m]['balanced_acc'] for m in W],labels=list(W),patch_artist=True)
    for p,m in zip(bp['boxes'],W): p.set_facecolor(W[m]); p.set_alpha(0.6)
    ax.axhline(0.5,color='k',ls=':',label='tebakan'); ax.set_ylabel("balanced accuracy"); ax.legend()
    ax.set_title(f"Sebaran antar-fold ({KELAS[1]} vs {KELAS[0]}, {SKENARIO})"); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"02_boxplot_balanced_acc.png"),dpi=150); plt.close()

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for a,nm in zip(axes,W):
        cm=confusion_matrix(oof[nm]['yt'],oof[nm]['yp']); a.imshow(cm,cmap='Blues')
        for (r,c),v in np.ndenumerate(cm): a.text(c,r,int(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
        a.set_title(nm); a.set_xticks([0,1]); a.set_yticks([0,1]); a.set_xticklabels(KELAS); a.set_yticklabels(KELAS)
        a.set_xlabel('prediksi'); a.set_ylabel('aktual')
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"03_confusion_matrix.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(len(ringkas))
    ax.bar(x,ringkas['balanced_acc_oof'],color=[W[m] for m in ringkas.model]); ax.axhline(0.5,color='grey',ls=':')
    for i,r in ringkas.iterrows(): ax.text(i,r['balanced_acc_oof']+0.01,f"p={r['p_value']}",ha='center',fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ringkas.model); ax.set_ylabel("balanced accuracy (OOF)"); ax.set_ylim(0,1)
    ax.set_title("Uji permutasi"); plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"04_uji_permutasi.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(8,6)); sizes=(ringkas['n_param']/ringkas['n_param'].max())*900+100
    for i,r in ringkas.iterrows():
        ax.scatter(r['latensi_ms'],r['balanced_acc_oof'],s=sizes[i],color=W[r['model']],alpha=0.75,edgecolor='k',zorder=3)
        ax.annotate(f"{r['model']}\n({r['n_param']:,} p)",(r['latensi_ms'],r['balanced_acc_oof']),xytext=(8,8),textcoords='offset points',fontsize=9)
    ax.axhline(0.5,color='grey',ls=':',label='tebakan'); ax.set_xlabel("latensi (ms/jendela)"); ax.set_ylabel("balanced accuracy")
    ax.set_title(f"Pareto Benchmarking ({SKENARIO})"); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DIR_RUN,"05_pareto_benchmarking.png"),dpi=150); plt.close()

    fig,ax=plt.subplots(figsize=(7,5)); r2=ringkas.sort_values('balanced_acc_oof',ascending=False)
    ax.bar(r2['model'],r2['balanced_acc_oof'],yerr=r2['balanced_acc_std'],color=[W[m] for m in r2.model],capsize=5)
    ax.axhline(0.5,color='k',ls=':'); ax.set_ylabel("balanced accuracy (OOF)"); ax.set_ylim(0,1); ax.set_title("Peringkat Model")
    plt.tight_layout(); plt.savefig(os.path.join(DIR_RUN,"06_peringkat_model.png"),dpi=150); plt.close()


if __name__=="__main__":
    main()
