# -*- coding: utf-8 -*-
"""
reproduce_published_numbers.py
==============================================================
Menghitung ulang setiap angka yang dilaporkan dalam naskah dari berkas CSV
turunan yang diterbitkan di Zenodo, lalu mencetak nilai hitungan di samping
nilai yang tercetak di naskah, sehingga setiap selisih langsung terlihat.

Skrip ini TIDAK melatih model dan TIDAK memerlukan TensorFlow maupun data EEG
mentah. Skrip ini hanya memverifikasi bahwa angka naskah benar-benar berasal
dari berkas hasil yang diterbitkan.

Cara pakai
----------
    pip install pandas numpy scipy
    python reproduce/reproduce_published_numbers.py

Secara bawaan skrip mencari berkas data di folder ./data relatif terhadap akar
repositori. Unduh paket Zenodo, buka isinya, lalu letakkan foldernya seperti
ini:

    eeg-vr-device-comparison/
      data/                      <- isi folder data/ dari paket Zenodo
        00_cohort_summary.csv
        01_participants.csv
        02_classification_cross_subject.csv
        03_classification_within_subject.csv
        04_classification_eye_state_control.csv
        05_paired_device_differences.csv
        06_selfreport_ssq_fms_sam.csv
        07_model_cost.csv
      reproduce/
        reproduce_published_numbers.py

Atau tunjuk foldernya langsung:

    python reproduce/reproduce_published_numbers.py --data /path/ke/folder/data
==============================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Lokasi data. Akar repositori adalah folder induk dari folder skrip ini.
# ---------------------------------------------------------------------------
AKAR = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", default=str(AKAR / "data"),
                help="folder yang memuat berkas CSV dari paket Zenodo")
arg = ap.parse_args()
DATA = Path(arg.data)

if not DATA.is_dir():
    sys.exit(f"Folder data tidak ditemukan: {DATA}\n"
             f"Unduh paket Zenodo, lalu jalankan ulang dengan --data <folder>.")

# ---------------------------------------------------------------------------
# Penghitung kecocokan. Setiap pemeriksaan mencetak satu baris.
# ---------------------------------------------------------------------------
cocok = 0
beda = 0
tandai = 0


def periksa(label, hitung, naskah, toleransi=0.05, satuan=""):
    """Bandingkan satu nilai hitungan dengan nilai yang tercetak di naskah."""
    global cocok, beda
    selisih = abs(hitung - naskah)
    status = "ok   " if selisih <= toleransi else "BEDA "
    if selisih <= toleransi:
        cocok += 1
    else:
        beda += 1
    print(f"  {status} {label:<44s} hitung={hitung:>8.2f}{satuan}"
          f"  naskah={naskah:>8.2f}{satuan}")


def catatan(label, isi):
    """Tandai nilai yang sengaja tidak dibandingkan, dengan alasannya."""
    global tandai
    tandai += 1
    print(f"  TANDAI {label}")
    for baris in isi.splitlines():
        print(f"         {baris}")


def muat(nama):
    """Baca satu berkas CSV dari folder data."""
    p = DATA / nama
    if not p.exists():
        sys.exit(f"Berkas hilang: {p}")
    return pd.read_csv(p)


print("=" * 78)
print("VERIFIKASI ANGKA NASKAH TERHADAP DATA TERBITAN")
print(f"folder data: {DATA}")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1. Kohort
# ---------------------------------------------------------------------------
print("\n1. KOHORT")
peserta = muat("01_participants.csv")
berpasangan = muat("05_paired_device_differences.csv")

periksa("jumlah peserta yang dianalisis", len(peserta), 58, toleransi=0)
periksa("peserta berpasangan dua perangkat",
        berpasangan.participant_id.nunique(), 56, toleransi=0)

# ---------------------------------------------------------------------------
# 2. Akurasi berimbang, tiga paradigma evaluasi
#    Nilai naskah: (CNN, CNN-LSTM, LSTM) untuk tiap perangkat, dalam persen.
# ---------------------------------------------------------------------------
NASKAH_AKURASI = {
    "02_classification_cross_subject.csv": {
        "label": "lintas-subjek",
        "four_channel": {"CNN": 59.0, "CNN-LSTM": 60.9, "LSTM": 61.8},
        "single_channel": {"CNN": 49.7, "CNN-LSTM": 49.8, "LSTM": 49.7},
    },
    "03_classification_within_subject.csv": {
        "label": "dalam-subjek",
        "four_channel": {"CNN": 61.3, "CNN-LSTM": 65.7, "LSTM": 66.0},
        "single_channel": {"CNN": 49.5, "CNN-LSTM": 50.4, "LSTM": 53.4},
    },
    "04_classification_eye_state_control.csv": {
        "label": "kontrol mata",
        "four_channel": {"CNN": 88.9, "CNN-LSTM": 90.0, "LSTM": 90.2},
        "single_channel": {"CNN": 71.8, "CNN-LSTM": 71.6, "LSTM": 69.1},
    },
}

NAMA_PERANGKAT = {"four_channel": "4 kanal", "single_channel": "1 kanal"}

print("\n2. AKURASI BERIMBANG (%)")
for berkas, acuan in NASKAH_AKURASI.items():
    d = muat(berkas)
    print(f"\n  -- {acuan['label']} --")
    # rerata akurasi berimbang per (perangkat, arsitektur), dihitung dari
    # nilai per-peserta yang diterbitkan
    rerata = d.groupby(["device", "model"])["balanced_accuracy"].mean() * 100
    for perangkat in ("four_channel", "single_channel"):
        for model in ("CNN", "CNN-LSTM", "LSTM"):
            periksa(f"{NAMA_PERANGKAT[perangkat]} {model}",
                    rerata.loc[(perangkat, model)],
                    acuan[perangkat][model], satuan="%")

# ---------------------------------------------------------------------------
# 3. Selisih berpasangan antar perangkat
#    Nilai naskah dalam poin persentase, beserta jumlah peserta yang unggul
#    pada perangkat 4 kanal.
# ---------------------------------------------------------------------------
NASKAH_SELISIH = {
    ("cross_subject", "CNN"): (8.8, 47),
    ("cross_subject", "CNN-LSTM"): (10.8, 46),
    ("cross_subject", "LSTM"): (11.6, 47),
    ("within_subject", "CNN"): (11.9, 40),
    ("within_subject", "CNN-LSTM"): (15.7, 48),
    ("within_subject", "LSTM"): (12.8, 42),
    ("eye_state_control", "CNN"): (16.8, 49),
    ("eye_state_control", "CNN-LSTM"): (18.2, 49),
    ("eye_state_control", "LSTM"): (20.9, 51),
}

print("\n3. SELISIH BERPASANGAN 4 KANAL - 1 KANAL (poin persentase)")
for (tugas, model), (sel_naskah, unggul_naskah) in NASKAH_SELISIH.items():
    g = berpasangan[(berpasangan.task == tugas) & (berpasangan.model == model)]
    periksa(f"{tugas} {model} selisih", g.difference.mean() * 100,
            sel_naskah, satuan=" pp")
    periksa(f"{tugas} {model} peserta unggul 4 kanal",
            (g.difference > 0).sum(), unggul_naskah, toleransi=0)

# uji Wilcoxon dua arah pada selisih berpasangan, seperti dinyatakan di naskah
print("\n   uji Wilcoxon berpasangan dua arah (p):")
for (tugas, model) in NASKAH_SELISIH:
    g = berpasangan[(berpasangan.task == tugas) & (berpasangan.model == model)]
    u = stats.wilcoxon(g.four_channel, g.single_channel)
    print(f"     {tugas:<18s} {model:<9s} n={len(g):>3d}  p={u.pvalue:.2e}")

# ---------------------------------------------------------------------------
# 4. Biaya model
# ---------------------------------------------------------------------------
print("\n4. BIAYA MODEL")
biaya = muat("07_model_cost.csv").set_index("model")
NASKAH_BIAYA = {
    "CNN": (18586, 1.14),
    "LSTM": (35777, 1.69),
    "CNN-LSTM": (11681, 4.01),
}
for model, (par, lat) in NASKAH_BIAYA.items():
    periksa(f"{model} jumlah parameter", biaya.loc[model, "parameters"],
            par, toleransi=0)
    periksa(f"{model} latensi inferensi",
            biaya.loc[model, "inference_latency_ms"], lat,
            toleransi=0.005, satuan=" ms")

# ---------------------------------------------------------------------------
# 5. Laporan diri SSQ dan FMS
#    Perhatian: berkas 06 memuat lebih dari satu sesi untuk sebagian peserta
#    (95 peserta-perangkat dengan 2 sesi, 5 dengan 3 sesi, 6 dengan 1 sesi),
#    sehingga nilai per peserta dirata-rata lebih dahulu sebelum diuji.
# ---------------------------------------------------------------------------
print("\n5. LAPORAN DIRI SSQ DAN FMS")
diri = muat("06_selfreport_ssq_fms_sam.csv")
kode56 = set(berpasangan.participant_id)
diri56 = diri[diri.participant_id.isin(kode56)]


def uji_berpasangan(d, kolom):
    """Rata-ratakan sesi per peserta, lalu uji Wilcoxon berpasangan."""
    lebar = d.pivot_table(index="participant_id", columns="device",
                          values=kolom).dropna()
    u = stats.wilcoxon(lebar.four_channel, lebar.single_channel)
    z = stats.norm.isf(u.pvalue / 2)          # ukuran efek r = z / sqrt(2n)
    r = z / np.sqrt(2 * len(lebar))
    return (lebar.four_channel.mean(), lebar.single_channel.mean(),
            u.pvalue, r, len(lebar))

m4, m1, p, r, n = uji_berpasangan(diri56, "ssq_total_after")
print(f"  SSQ pasca-sesi, 56 peserta berpasangan: "
      f"4 kanal={m4:.2f}  1 kanal={m1:.2f}  n={n}  p={p:.3f}  r={r:.3f}")
periksa("SSQ pasca-sesi 4 kanal", m4, 24.68, toleransi=0.5)
periksa("SSQ pasca-sesi 1 kanal", m1, 11.26, toleransi=0.5)
periksa("SSQ pasca-sesi nilai p", p, 0.006, toleransi=0.005)

m4b, m1b, pb, rb, nb = uji_berpasangan(diri56, "ssq_total_before")
print(f"  SSQ pra-sesi, 56 peserta berpasangan:   "
      f"4 kanal={m4b:.2f}  1 kanal={m1b:.2f}  n={nb}  p={pb:.3f}")
catatan("SSQ pra-sesi",
        f"Naskah menyatakan 6.23 lawan 3.57 dengan p = 0.044, dan menyimpulkan\n"
        f"sebagian selisih sudah ada sebelum sesi. Dari data terbitan nilainya\n"
        f"{m4b:.2f} lawan {m1b:.2f} dengan p = {pb:.3f}, yang tidak signifikan.\n"
        f"Kalimat naskah tentang selisih pra-sesi perlu diperiksa ulang sebelum\n"
        f"resubmisi. Lihat docs/known_discrepancies.md.")

m4c, m1c, pc, rc, nc = uji_berpasangan(diri56, "fms_peak")
print(f"  FMS puncak, 56 peserta berpasangan:     "
      f"4 kanal={m4c:.2f}  1 kanal={m1c:.2f}  n={nc}  p={pc:.3f}")
periksa("FMS puncak nilai p tidak signifikan", 1.0 if pc > 0.05 else 0.0,
        1.0, toleransi=0)

# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"RINGKASAN: {cocok} cocok, {beda} berbeda, {tandai} ditandai untuk ditelaah")
print("=" * 78)
sys.exit(0 if beda == 0 else 1)
