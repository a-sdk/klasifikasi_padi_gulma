# Program Klasifikasi Padi dan Gulma

## Deskripsi Proyek

Proyek ini adalah program untuk mengklasifikasikan padi dan gulma pada citra hasil tangkapan drone multispektral.

## Fitur Utama

1. Membuat tumpukan fitur yang terdiri dari kanal (band) citra, transformasi indeks vegetasi, dan tekstur (opsional).
2. Membuat data latihan untuk melatih model random forest berdasarkan tumpukan fitur.
3. Melakukan segmentasi dan klasifikasi padi dan gulma menggunakan model random forest.
4. Memeriksa ukuran raster.
5. Melakukan ekstraksi nilai piksel atau fitur berdasarkan shapefile poligon.


## Panduan Instalasi dan Penggunaan

Ikuti langkah-langkah berikut untuk menjalankan proyek di komputer lokal Anda.

### Persyaratan

Pastikan Anda telah menginstal Python 3.11 di komputer Anda.

### Instalasi

Salin (clone) atau unduh repositori ini ke folder lokal Anda.

Buka terminal atau Command Prompt dan navigasi ke folder proyek.

### Menyiapkan Virtual Environment

Jalankan perintah berikut untuk membuat virtual environment. Ganti `<nama_virtual_environment>` dengan nama yang diinginkan.

    py -m venv <nama_virtual_environment>

### Menginstall Libraries

Jalankan perintah berikut untuk menginstal semua libraries (pustaka) yang diperlukan:

    pip install -r requirements.txt

Catatan: Pastikan `requirements.txt` sudah ada di folder Anda.


### Menjalankan Program

Jalankan perintah berikut di terminal Anda:

    py program.py

Program akan berjalan dan menunjukkan status proses yang dilakukan.


## Struktur Proyek

    /klasifikasi_padi_gulma
    ├── cek_ukuran_raster.py            # Program untuk memeriksa ukuran citra.
    ├── data_latih                      # shapefile berisi label untuk padi dan gulma.
    ├── ekstrak_fitur.py                # Program untuk mengekstrak fitur.
    ├── latih_model.py                  # Program untuk melatih model random forest.
    ├── model_random_forest_0.joblib    # Model RF yang dilatih berdasarkan 11 fitur.
    ├── model_random_forest_1.joblib    # Model RF yang dilatih berdasarkan 13 fitur.
    ├── segmentasi_gulma.py             # Program untuk menjalankan klasifikasi padi dan gulma.
    └── tumpuk_fitur.py                 # Program untuk menumpuk seluruh fitur.


---

## Changelogs
### Versi 1.0 - 31 Oktober 2025

* Rilis awal program klasifikasi padi dan gulma.

