import re
import warnings
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

DATASET_FILE = "krl_clean.csv"   
STOPWORDS_FILE = "stopwords_krl.txt"
MAX_DF = 0.4             
MIN_DF = 2                
JUMLAH_KATA_DOMINAN = 5
JUMLAH_CONTOH = 3
TOP_WORD_COUNT = 10
MAX_WORDS_CLOUD = 60
PETA_WARNA = ["Blues", "Oranges", "Greens", "Purples", "Reds"]

# Nama tema ditentukan setelah membaca kata dominan dan contoh postingan tiap cluster.
TEMA_CLUSTER = {
    0: "Kronologi Kecelakaan dan Peran Taksi di Perlintasan",
    1: "Korban Jiwa, Perawatan, dan Ungkapan Duka",
    2: "Usulan Gerbong Khusus Perempuan oleh Menteri PPPA",
}
# Ubah menjadi True bila kolom tema_cluster di CSV sudah diperbarui dan ingin dipakai langsung.
PAKAI_TEMA_DARI_CSV = False


def cari_berkas(nama):
    """Cari berkas di folder halaman ini, folder induknya, atau subfolder data/."""
    folder = Path(__file__).resolve().parent
    kandidat = [
        folder / nama, folder.parent / nama,
        folder / "data" / nama, folder.parent / "data" / nama,
        Path.cwd() / nama, Path.cwd() / "data" / nama,
    ]
    for p in kandidat:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Menghitung kata dominan, word count, dan word cloud...")
def hitung_clustering(path_data, waktu_data, path_sw, waktu_sw):
    """Membaca dataset dan menghitung semua keluaran halaman.

    Label cluster diambil dari kolom `cluster` di CSV (hasil K-Means di notebook).
    Kata dominan dan contoh postingan dihitung dari TF-IDF dengan parameter yang sama
    seperti notebook. Pusat cluster = rata-rata vektor TF-IDF anggotanya, yang sama
    dengan centroid K-Means yang sudah konvergen.
    waktu_data dan waktu_sw ikut menjadi kunci cache agar berkas yang diganti dibaca ulang.
    """
    df = pd.read_csv(path_data, encoding="utf-8-sig")
    stop_list = [w.strip() for w in Path(path_sw).read_text(encoding="utf-8").splitlines() if w.strip()]
    stop_set = set(stop_list)

    kolom_teks = "full_text" if "full_text" in df.columns else ("text" if "text" in df.columns else None)
    hilang = [k for k in ("clean_text", "cluster") if k not in df.columns]
    if kolom_teks is None:
        hilang.append("full_text")
    if hilang:
        raise ValueError("Kolom wajib tidak tersedia di dataset: " + ", ".join(hilang))

    n_awal = len(df)
    df = df.dropna(subset=["clean_text", "cluster"]).copy()
    df["clean_text"] = df["clean_text"].astype(str)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    df["cluster"] = df["cluster"].astype(int)
    if len(df) < 10 or df["cluster"].nunique() < 2:
        raise ValueError("Data valid terlalu sedikit atau hanya memiliki satu cluster.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = TfidfVectorizer(stop_words=stop_list, max_df=MAX_DF, min_df=MIN_DF)
        X = vec.fit_transform(df["clean_text"])
    terms = vec.get_feature_names_out()

    hasil = {}
    for k in sorted(df["cluster"].unique()):
        idx = np.where(df["cluster"].to_numpy() == k)[0]
        pusat = np.asarray(X[idx].mean(axis=0)).ravel()

        kata_dominan = [terms[j] for j in pusat.argsort()[::-1][:JUMLAH_KATA_DOMINAN]]
        kemiripan = np.asarray(X[idx].dot(pusat)).ravel()
        contoh = [df.loc[i, kolom_teks] for i in idx[np.argsort(-kemiripan)[:JUMLAH_CONTOH]]]

        teks_cluster = " ".join(df.loc[idx, "clean_text"])
        word_count = Counter(w for w in teks_cluster.split() if w not in stop_set and len(w) > 2)

        try:
            # collocations=False: hitung per kata (bukan pasangan kata) supaya sesuai tabel word count.
            wc = WordCloud(
                stopwords=stop_set, background_color="white",
                max_words=MAX_WORDS_CLOUD, width=800, height=500, random_state=42,
                collocations=False,
            ).generate(teks_cluster)
            peta_warna = plt.get_cmap(PETA_WARNA[int(k) % len(PETA_WARNA)])
            frek_relatif = wc.words_   # frekuensi relatif tiap kata (kata terbanyak = 1.0)

            def warna_kata(word, **kwargs):
                # Makin sering muncul, makin gelap. Rentang 0.35-1.0 dipakai agar tidak ada kata yang terlalu pudar.
                t = 0.35 + 0.65 * (frek_relatif.get(word, 0) ** 0.5)
                r, g, b, _ = peta_warna(t)
                return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

            gambar_wc = wc.recolor(color_func=warna_kata).to_array()
        except ValueError:
            gambar_wc = None

        hasil[int(k)] = {
            "jumlah": int(len(idx)),
            "kata_dominan": kata_dominan,
            "contoh": contoh,
            "word_count": word_count.most_common(TOP_WORD_COUNT),
            "wordcloud": gambar_wc,
        }

    sil = float(silhouette_score(X, df["cluster"])) if df["cluster"].nunique() >= 2 else None
    tema_csv = {}
    if "tema_cluster" in df.columns:
        tema_csv = {int(k): str(v) for k, v in df.groupby("cluster")["tema_cluster"].first().items()}
    return hasil, sil, n_awal - len(df), len(df), tema_csv


def aman(teks):
    """Buang URL dan lepas karakter markdown supaya teks tampil apa adanya."""
    teks = re.sub(r"https?://\S+", "", str(teks)).strip()
    return re.sub(r"([\\`*_{}\[\]#+!|>~$])", r"\\\1", teks)


# ------------------------------- Halaman -------------------------------
st.title("🧩 Clustering Teks")
st.caption("Hasil TF-IDF + K-Means dengan 3 cluster dari notebook kelompok. Label cluster diambil dari "
           "dataset; kata dominan dan contoh postingan dihitung dengan parameter yang sama seperti notebook.")

path_data = cari_berkas(DATASET_FILE)
path_sw = cari_berkas(STOPWORDS_FILE)
if path_data is None:
    st.error(f"Dataset '{DATASET_FILE}' tidak ditemukan. Letakkan berkas itu di folder aplikasi atau di "
             "subfolder data/, atau sesuaikan DATASET_FILE di kode.")
    st.stop()
if path_sw is None:
    st.error(f"Berkas daftar stopword '{STOPWORDS_FILE}' tidak ditemukan. Letakkan berkas itu di folder "
             "yang sama dengan dataset.")
    st.stop()

try:
    hasil, sil, n_keluar, total, tema_csv = hitung_clustering(
        str(path_data), path_data.stat().st_mtime, str(path_sw), path_sw.stat().st_mtime
    )
except Exception as e:
    st.error(f"Data tidak dapat diproses untuk clustering: {e}")
    st.stop()

ks = sorted(hasil)


def nama_tema(k):
    if PAKAI_TEMA_DARI_CSV and tema_csv.get(k):
        return tema_csv[k]
    return TEMA_CLUSTER.get(k) or tema_csv.get(k) or f"Cluster {k}"


if n_keluar > 0:
    st.info(f"{n_keluar} baris dikeluarkan dari clustering karena teks bersih atau label cluster kosong.")

# --- Ringkasan semua cluster ---
st.subheader("Ringkasan cluster")
c1, c2 = st.columns(2)
c1.metric("Jumlah postingan", f"{total:,}")
c2.metric("Jumlah cluster", len(ks))

ringkasan = pd.DataFrame({
    "Cluster": ks,
    "Tema": [nama_tema(k) for k in ks],
    "Jumlah postingan": [hasil[k]["jumlah"] for k in ks],
    "Persentase": [f"{hasil[k]['jumlah'] / total * 100:.1f}%" for k in ks],
    "Lima kata dominan": [", ".join(hasil[k]["kata_dominan"]) for k in ks],
})
st.dataframe(ringkasan, hide_index=True)
st.bar_chart(pd.Series({f"Cluster {k}": hasil[k]["jumlah"] for k in ks}, name="Jumlah postingan"))

# --- Rincian tiap cluster ---
st.subheader("Rincian tiap cluster")
tabs = st.tabs([f"Cluster {k}" for k in ks])
for i, (k, tab) in enumerate(zip(ks, tabs)):
    h = hasil[k]
    with tab:
        st.markdown(f"### Cluster {k}: {nama_tema(k)}")
        m1, m2 = st.columns(2)
        m1.metric("Jumlah postingan", f"{h['jumlah']:,}")
        m2.metric("Persentase dari seluruh data", f"{h['jumlah'] / total * 100:.1f}%")

        st.markdown("**Lima kata dominan** (bobot TF-IDF tertinggi pada pusat cluster)")
        st.markdown(", ".join(f"`{w}`" for w in h["kata_dominan"]))

        st.markdown("**Tiga contoh postingan** (paling dekat dengan pusat cluster)")
        for n, teks in enumerate(h["contoh"], 1):
            st.markdown(f"{n}. {aman(teks)}")

        st.markdown(f"**Word count: {TOP_WORD_COUNT} kata paling sering muncul**")
        if h["word_count"]:
            kata_list, hitung_list = zip(*h["word_count"])
            kiri, kanan = st.columns([1, 2])
            kiri.dataframe(pd.DataFrame({"Kata": kata_list, "Frekuensi": hitung_list}), hide_index=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(kata_list[::-1], hitung_list[::-1], color=plt.cm.tab10(i % 10))
            for bar in bars:
                ax.text(bar.get_width() + max(hitung_list) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{int(bar.get_width())}", va="center", ha="left", fontsize=9)
            ax.set_xlabel("Frekuensi kemunculan")
            ax.set_title(f"Top {TOP_WORD_COUNT} word count: Cluster {k}")
            kanan.pyplot(fig)
            plt.close(fig)
            st.caption("Dihitung dari teks bersih, setelah stopword dan kata dua huruf atau kurang dibuang.")
        else:
            st.info("Tidak ada kata yang dapat dihitung pada cluster ini.")

        st.markdown("**Word cloud**")
        if h["wordcloud"] is not None:
            st.image(h["wordcloud"])
        else:
            st.info("Tidak cukup kata untuk membuat word cloud pada cluster ini.")
