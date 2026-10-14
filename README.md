# Final Project: Analisis Teks, Clustering, Regresi & Analisis Sentimen Postingan X (Twitter) Kasus KRL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1paPq5ZOwojkOpMj9AlhIYBcWmelSCK-B?usp=sharing)

**Mata Kuliah:** Praktik Aplikasi Web  
**Kelas:** I2  
**Kelompok:** 4  
**Topik Kasus:** Tragedi & Isu Layanan KRL Commuter Line Bekasi Timur (April 2026)  
**Notebook Google Colab:** [Buka Notebook di Google Colab](https://colab.research.google.com/drive/1paPq5ZOwojkOpMj9AlhIYBcWmelSCK-B?usp=sharing)

---

## 👥 Anggota Kelompok & Pembagian Tugas

| No | Nama Anggota | Bagian Tugas | Rincian Tanggung Jawab & Kontribusi |
|:---:|:---|:---:|:---|
| 1 | **Redina** | **Bagian 1 & 5** | • **Bagian 1:** Penyusunan Stopwords (Bahasa Indonesia & Inggris), kata gaul/slang media sosial, dan filter proteksi kata negasi (*tidak, belum, jangan, gak*).<br>• **Bagian 5:** Visualisasi Reduksi Dimensi dengan PCA 2D (Principal Component Analysis) untuk sebaran cluster data. |
| 2 | **Salya** | **Bagian 3 & 4** | • **Bagian 3:** Analisis Karakteristik Cluster, ekstraksi top 10 kata dominan centroid, contoh tweet representatif, dan ukuran tiap cluster.<br>• **Bagian 4:** Evaluasi Kualitas & Kejelasan Cluster (Silhouette Score) serta Uji Stabilitas Pengelompokan (Adjusted Rand Index / ARI). |
| 3 | **Alvito** | **Bagian 2 & 6 + Sentimen + README** | • **Bagian 2:** Ekstraksi Fitur Teks dengan TF-IDF Vectorizer & Pemodelan K-Means ($k=3$).<br>• **Bagian 6:** Penamaan Tema Cluster, **Word Cloud per Cluster (Saran Dosen)**, Visualisasi Word Count Tema.<br>• **Tahap Sentimen:** Pemodelan Analisis Sentimen (Positif, Netral, Negatif), visualisasi proporsi sentimen per cluster, ekspor dataset final, dan dokumentasi README.md. |

*(Catatan: Terdapat juga modul tambahan **Regresi Linier** untuk prediksi metrik interaksi postingan/likes berdasarkan jumlah kata dan hashtag).*

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis opini publik, respons netizen, dan klasterisasi isu di media sosial X (Twitter) terkait insiden dan dinamika layanan KRL Commuter Line di perlintasan Stasiun Bekasi Timur pada akhir April 2026. 

Alur analisis menggabungkan:
1. **Unsupervised Learning (K-Means Clustering):** Memetakan 3 tema sentral pembahasan masyarakat.
2. **Word Cloud & Word Count per Cluster:** Visualisasi topik dominan per tema berdasarkan saran dosen pembimbing.
3. **Regresi Linier:** Eksperimen prediksi likes (*favorite count*) berdasarkan karakteristik panjang teks dan jumlah hashtag.
4. **Sentiment Analysis:** Mengukur proporsi sentimen (Positif, Netral, Negatif) pada tiap kelompok klaster pembahasan.
5. **Aplikasi Web Interaktif (Streamlit):** Dashboard eksplorasi data, hasil klasterisasi, regresi, dan sentimen secara visual.

---

## 🔄 Alur & Arsitektur Analisis

```
[Crawling X (tweet-harvest)] 
          ↓ 860 Tweet Mentah
[Sanitasi & Data Cleaning]  → Normalisasi Newline, Hapus Mojibake/Karakter Asing, Quoting CSV
          ↓
[Bagian 1: Stopwords & Negasi (Redina)]
          ↓
[Bagian 2: TF-IDF & K-Means (Alvito)]  → 3 Cluster Terbentuk (k=3)
          ↓
[Bagian 3: Karakteristik Cluster (Salya)]  → Top Words & Sampel Tweet Centroid
          ↓
[Bagian 4: Silhouette Score & ARI (Salya)]  → Uji Kohesi & Stabilitas Clustering
          ↓
[Bagian 5: Visualisasi PCA 2D (Redina)]  → Scatter Plot 2 Dimensi Persebaran Data
          ↓
[Bagian 6: Tema, Word Cloud & Ekspor (Alvito)]  → Word Cloud per Tema (Saran Dosen)
          ↓
[Modul Tambahan: Regresi Linier]  → Prediksi Likes vs Jumlah Kata & Hashtag
          ↓
[Bagian 7: Analisis Sentimen (Alvito)]  → Klasifikasi Polaritas (Positif, Netral, Negatif) per Cluster
```

---

## 📊 Ringkasan Hasil Analisis

### 1. Hasil Clustering ($k=3$)
Berdasarkan pembobotan TF-IDF dan K-Means, diperoleh 3 tema sentral:
* **Cluster 0 (60.55% | 502 tweet):** *Respons Kebijakan & Informasi Operasional KRL*
  * Kata Kunci Dominan: `kereta`, `taksi`, `cikarang`, `anggrek`, `sm`, `sopir`, `polisi`, `pmi`
  * Fokus: Kronologi tertempernya taksi di perlintasan, respons kepolisian/Dishub, dan apresiasi relawan PMI oleh Pak JK.
* **Cluster 1 (32.81% | 272 tweet):** *Keterlambatan, Gangguan Jalur & Keluhan Penumpang*
  * Kata Kunci Dominan: `korban`, `meninggal`, `anggrek`, `keluarga`, `luka`, `duka`, `bunga`, `pekerja`
  * Fokus: Ungkapan duka cita mendalam bagi korban jiwa di Stasiun Bekasi Timur serta aksi solidaritas karangan bunga.
* **Cluster 2 (6.63% | 55 tweet):** *Insiden, Evakuasi & Dampak Perjalanan Stasiun*
  * Kata Kunci Dominan: `gerbong`, `wanita`, `perempuan`, `menteri`, `khusus`, `hancur`, `terhimpit`
  * Fokus: Sorotan kondisi gerbong khusus wanita pasca insiden dan evaluasi keselamatan dari Kementerian Perhubungan.

### 2. Evaluasi Validitas Cluster
* **Silhouette Score:** Nilai pada teks Twitter bernilai $0.0148$ (khas matriks teks sparse berdimensi tinggi).
* **Stabilitas Pengelompokan (ARI):** Nilai konsistensi berkisar antara **0.818 s.d. 0.934** (mendekati 1.0) saat diuji dengan berbagai *random seed*, membuktikan pengelompokan K-Means sangat stabil dan konsisten.

### 3. Analisis Sentimen
* Sentimen dominan pada percakapan seputar tragedi KRL didominasi oleh respons emosional negatif (duka, korban, kelalaian perlintasan) serta sentimen netral (pemberitaan/informasi jalur), dengan sebagian sentimen positif berfokus pada kesigapan evakuasi PMI dan solidaritas warga.

---

## 🛠️ Petunjuk Menjalankan

### A. Menjalankan Notebook di Google Colab
Notebook lengkap yang sudah tervalidasi dapat langsung diakses dan dijalankan melalui tautan resmi Google Colab berikut:
👉 **[Buka di Google Colab](https://colab.research.google.com/drive/1paPq5ZOwojkOpMj9AlhIYBcWmelSCK-B?usp=sharing)**

1. Klik tombol **Open in Colab** di atas.
2. Gunakan runtime Python 3 (CPU standar).
3. Anda dapat langsung menjalankan seluruh sel secara berurutan. Jika runtime baru dibuka dan ingin langsung menganalisis sentimen/clustering tanpa mengulang crawling, cukup gunakan dataset `hasil_clustering_krl_clean_1000.csv` yang otomatis tersedia di repositori ini.

### B. Menjalankan Aplikasi Web (Streamlit Dashboard)
Untuk menjalankan dashboard interaktif berbasis Streamlit secara lokal:

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan aplikasi web
streamlit run app.py
```

Dashboard akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 📁 Struktur File Repositori

```
final-project/
│
├── pages/                                # Halaman Dashboard Streamlit
│   ├── clustering.py                     # Visualisasi K-Means, PCA & Word Cloud
│   ├── data.py                           # Pratinjau dataset & ringkasan
│   ├── regresi.py                        # Model prediksi likes postingan
│   └── sentimen.py                       # Visualisasi analisis sentimen
├── app.py                                # Entry point aplikasi web Streamlit
├── common.py                             # Fungsi utilitas pemrosesan data & kamus
├── requirements.txt                      # Dependensi pustaka Python
├── stopwords_krl.txt                     # Daftar stopwords khusus
├── model_regresi_likes.pkl               # Model Regresi Linier terlatih
│
├── Kelompok_4_I2.ipynb                   # Notebook UTAMA lengkap (Crawling s.d. Analisis Sentimen)
├── hasil_analisis_sentimen_krl.csv       # Dataset final berlabel cluster & sentimen (829 baris)
├── hasil_clustering_krl_clean_1000.csv   # Dataset bersih berlabel cluster (829 baris)
├── krl_clean.csv                         # Dataset bersih untuk aplikasi web
├── krl_tweets_1000.csv                   # Data mentah crawling tweet-harvest
├── ringkasan_cluster_krl.csv             # Ringkasan persentase dan kata dominan tiap cluster
├── Proposal Aplikasi Web...pdf           # Dokumen proposal final project kelompok
└── README.md                             # Dokumentasi teknis proyek
```

---
*Dibuat untuk memenuhi tugas Final Project Praktik Aplikasi Web Kelas I2 - 2026.*
