# Catatan menjalankan skrip analisis

Keenam skrip di folder ini adalah skrip **sebagaimana dijalankan** saat
penelitian, termasuk jalur Windows yang dipakai. Skripnya tidak dirapikan
ulang, supaya kode yang diterbitkan benar-benar kode yang menghasilkan angka
di naskah.

## Satu hal yang harus diubah

Setiap skrip punya tetapan `BASE_DIR` di bagian atas, misalnya:

```python
BASE_DIR = r"D:\python vscode\venv\14_klasifikasieeg_muse_2026"
```

Ubah nilainya ke folder yang memuat berkas rekaman per detik yang sudah
digabung (`MASTER_GABUNGAN.csv` untuk perangkat 4 kanal, berkas padanannya
untuk perangkat 1 kanal). Tidak ada lagi yang perlu diubah.

Berkas rekaman per detik tidak diterbitkan. Lihat `docs/data_access.md`.

## Pasangan skrip

| Paradigma | 4 kanal | 1 kanal |
|---|---|---|
| Lintas-subjek | `04_klasifikasi_cnn_lstm.py` | `04_muse_klasifikasi.py` |
| Dalam-subjek | `04b_klasifikasi_within_subject.py` | `04b_muse_klasifikasi.py` |
| Kontrol mata | `04c_klasifikasi_task_baseline.py` | `04c_muse_klasifikasi.py` |

Tiap pasangan memakai window, normalisasi, benih pembagian data, jumlah epoch,
ukuran batch, laju belajar, bobot kelas dan callback yang **identik**. Hanya
fungsi `build_*` yang berbeda antar arsitektur. Itulah yang membuat
perbandingan antar arsitektur setara.

## Tombol di dalam skrip

Skrip perangkat 1 kanal punya tetapan `SKENARIO`:

- `"SEMUA"` — 20 daya pita empat kanal ditambah asimetri frontal
- `"TANPA_FRONTAL"` — hanya kanal temporal TP9/TP10, untuk menguji dampak
  cemaran EMG frontal

Jalankan dua kali dengan nilai berbeda untuk membandingkan. Skrip `04c` punya
tetapan `TARGET` dengan pilihan `"MATA"` (kontrol positif utama) dan
`"TASK_BASELINE"`.

Tetapan `QUICK_TEST` memangkas jumlah epoch untuk uji jalan cepat. Pastikan
nilainya `False` saat menjalankan untuk hasil sebenarnya.

## Waktu jalan

Pada CPU, satu skrip lintas-subjek dengan LOSO atas 59 peserta dan tiga
arsitektur memerlukan beberapa jam. Keluaran ditulis ke folder bertimestamp
berisi tabel nilai per peserta, grafik analisis dan grafik Pareto.
