# Final Project: Analisis Teks, Clustering & Analisis Sentimen Postingan X (Twitter) Kasus KRL

**Mata Kuliah:** Praktik Aplikasi Web  
**Kelas:** I2  
**Kelompok:** 4  
**Topik Kasus:** Tragedi & Isu Layanan KRL Commuter Line Bekasi Timur (April 2026)

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

Proyek ini bertujuan untuk menganalisis opini, respon publik, dan isu dominan di media sosial X (Twitter) terkait insiden dan dinamika layanan KRL Commuter Line di perlintasan Stasiun Bekasi Timur pada akhir April 2026. 

Alur analisis menggabungkan teknik **Unsupervised Learning (K-Means Clustering)** untuk memetakan tema pembahasan publik dan **Sentiment Analysis** untuk menilai polaritas respon masyarakat (positif, netral, atau negatif) pada tiap kelompok isu.

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
[Bagian 7: Analisis Sentimen (Alvito)]  → Klasifikasi Polaritas (Positif, Netral, Negatif) per Cluster
```

---

## 📊 Ringkasan Hasil Analisis

### 1. Hasil Clustering ($k=3$)
Berdasarkan pembobotan TF-IDF dan K-Means, diperoleh 3 tema sentral:
* **Cluster 0 (60.55% | 502 tweet):** *Respons Kebijakan & Informasi Operasional KRL*
  * Kata Kunci Dominan: `kereta`, `taksi`, `cikarang`, `anggrek`, `sm`, `sopir`, `polisi`, `pmi`
  * Fokus: Pemberitaan kronologi taksi Bluebird/SM yang tertemper, tindakan kepolisian, respon Dishub, dan apresiasi relawan PMI oleh Pak JK.
* **Cluster 1 (32.81% | 272 tweet):** *Keterlambatan, Gangguan Jalur & Keluhan Penumpang*
  * Kata Kunci Dominan: `korban`, `meninggal`, `anggrek`, `keluarga`, `luka`, `duka`, `bunga`, `pekerja`
  * Fokus: Ungkapan belasungkawa warga/pekerja atas jatuhnya korban jiwa di Stasiun Bekasi Timur serta karangan bunga duka cita.
* **Cluster 2 (6.63% | 55 tweet):** *Insiden, Evakuasi & Dampak Perjalanan Stasiun*
  * Kata Kunci Dominan: `gerbong`, `wanita`, `perempuan`, `menteri`, `khusus`, `hancur`, `terhimpit`
  * Fokus: Pembahasan kondisi gerbong khusus perempuan pasca tabrakan dan evaluasi keselamatan dari Kementerian Perhubungan.

### 2. Evaluasi Validitas Cluster
* **Silhouette Score:** Evaluasi kohesi antar data teks percakapan Twitter.
* **Stabilitas Pengelompokan (ARI):** Nilai konsistensi berkisar antara **0.818 s.d. 0.934** (mendekati 1.0) saat diuji dengan berbagai *random seed* berbeda, membuktikan cluster sangat stabil dan tidak acak.

---

## 🛠️ Petunjuk Menjalankan Notebook di Google Colab

1. **Buka Notebook:** Upload dan buka file `Kelompok_4_I2.ipynb` di [Google Colab](https://colab.research.google.com/).
2. **Koneksi Runtime:** Pilih runtime Python 3 (CPU standar sudah cukup).
3. **Eksekusi:**
   * Jika ingin menjalankan dari awal (*fresh run*), jalankan cell secara berurutan dari atas ke bawah.
   * **Tips Crawling:** Karena batasan *rate limit* gratisan dari Twitter/X, proses crawling akan mengumpulkan sekitar 860 tweet. Jika cell crawling berhenti atau muncul notifikasi *rate limit*, cukup klik tombol **Stop** pada cell tersebut dan langsung lanjutkan mengeksekusi cell berikutnya.
   * Jika hanya ingin melihat visualisasi dan analisis sentimen, cukup gunakan file dataset bersih `hasil_clustering_krl_clean_1000.csv` yang sudah disediakan di repositori.

---

## 📁 Struktur File Repositori

```
final-project/
│
├── Kelompok_4_I2.ipynb               # Notebook utama (Crawling, Clustering, Regresi & Sentimen)
├── hasil_clustering_krl_clean_1000.csv# Dataset bersih berlabel cluster (829 tweet)
├── ringkasan_cluster_krl.csv         # Ringkasan persentase dan kata dominan tiap cluster
├── krl_tweets_1000.csv               # Data mentah hasil crawling tweet-harvest
├── Proposal Aplikasi Web...pdf       # Dokumen proposal final project kelompok
└── README.md                         # Dokumentasi teknis proyek
```

---
*Dibuat untuk memenuhi tugas Final Project Praktik Aplikasi Web Kelas I2 - 2026.*

