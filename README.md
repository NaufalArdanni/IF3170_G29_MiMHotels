# Tugas Besar 2 (Supervised Learning) IF3170 Inteligensi Artifisial Semester 1 2026/2027 - Kelompok MiMHotels (29)

## Deskripsi Singkat

Repositori ini berisi pekerjaan kelompok MiMHotels (29) untuk Tugas Besar 2 IF3170 Inteligensi Artifisial Semester 1 2026/2027. Studi kasusnya adalah membantu **Ronaldo Hotel™** mengestimasi pengunjung yang akan membatalkan pesanannya, yaitu mengklasifikasikan target `is_canceled` (1 jika pemesanan dibatalkan, 0 jika tidak) dari data pemesanan penginapan. Kinerja model dinilai dengan **Macro F1-score** pada [kompetisi Kaggle](https://www.kaggle.com/competitions/T2-IF3170-2026-2027).

Algoritma pemelajaran mesin berikut diimplementasikan *from scratch* (hanya menggunakan pustaka untuk perhitungan matematika seperti NumPy) dan dibandingkan dengan implementasi *scikit-learn*:

1. K-Nearest Neighbors (KNN)
2. Decision Tree Learning (ID3, C4.5, atau CART)
3. Logistic Regression dengan regularisasi L1, L2, dan Elastic Net
4. Support Vector Machine (SVM) dengan kernel Linear, Polynomial, dan Gaussian
5. Bayesian Network (Naive Bayes)

Struktur Repositori:

```text
src/
  notebook.ipynb            notebook pengerjaan
  ronaldohotel/
    scratch/                implementasi algoritma from scratch
    baselines/              implementasi scikit-learn
    data/                   pemuatan, cleaning, dan preprocessing data
    evaluation/             metrics dan validasi
    plotting/               fungsi visualisasi
tests/                      pengujian otomatis
docs/                       laporan (.pdf)
output/                     hasil run algoritma
data/                       dataset kompetisi (tidak di-commit)
```

## Cara Setup dan Menjalankan Program

### Prasyarat

Instal **Git**, **uv**, dan **Visual Studio Code**. Python tidak perlu dipasang secara manual karena uv akan mengunduh versi yang dibutuhkan proyek ini (Python 3.13).

Cara Menginstal uv:

- Windows (PowerShell): `winget install --id=astral-sh.uv -e`
- macOS atau Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Tutup dan buka kembali terminal, lalu pastikan `uv --version` dapat dijalankan.

### Setup

1. *Clone* repositori dan masuk ke foldernya.

   ```bash
   git clone https://github.com/NaufalArdanni/IF3170_G29_MiMHotels.git
   cd IF3170_G29_MiMHotels
   ```

2. Pasang lingkungan sesuai versi yang terkunci pada `uv.lock`, lalu aktifkan Git hooks.

   ```bash
   uv sync --locked
   uv run pre-commit install
   ```

3. Unduh `train.csv`, `test.csv`, dan `sample_submission.csv` dari halaman kompetisi Kaggle, lalu letakkan pada folder `data/`.

4. Buka folder proyek di VS Code (`code .`), instal ekstensi yang direkomendasikan, lalu pilih *interpreter* dari `.venv` (`Ctrl+Shift+P`, lalu `Python: Select Interpreter`).

### Menjalankan Program

1. Buka `src/notebook.ipynb` di VS Code.
2. Pilih *kernel* dari `.venv`.
3. Jalankan seluruh sel dari atas ke bawah (*Run All*).

Hasil *run* disimpan pada folder `output/`, termasuk berkas submisi Kaggle.

Untuk menjalankan pengujian otomatis:

```bash
uv run pytest
```

## Pembagian Tugas

| Nama | NIM | Akun GitHub | Tugas |
| ---- | --- | ----------- | ----- |
| Naufal Baldemar Ardanni | 13521154 | NaufalArdanni | TODO |
| Theodore Marvel Suranton | 10623061 | TODO | TODO |
| Muhammad Iqbal Raihan | 13524011 | TODO | TODO |
| Jingglang Galih Rinenggan | 13524095 | TODO | TODO |
