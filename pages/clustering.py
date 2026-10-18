import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

DATASET_FILE = "krl_clean.csv"
STOPWORDS_FILE = "stopwords_krl.txt"
WARNA = ["Blues", "Oranges", "Greens"]   # satu warna word cloud untuk tiap cluster

# Nama tema ditentukan setelah membaca kata dominan dan contoh postingan.
TEMA = {
    0: "Kronologi Kecelakaan dan Peran Taksi di Perlintasan",
    1: "Korban Jiwa, Perawatan, dan Ungkapan Duka",
    2: "Usulan Gerbong Khusus Perempuan oleh Menteri PPPA",
}


def cari_berkas(nama):
    """Cari berkas di folder halaman ini, folder induknya, atau subfolder data/."""
    base = Path(__file__).resolve().parent
    for folder in (base, base.parent, base / "data", base.parent / "data"):
        if (folder / nama).exists():
            return folder / nama


def aman(teks):
    """Buang URL dan lepas karakter markdown supaya teks tampil apa adanya."""
    teks = re.sub(r"https?://\S+", "", str(teks)).strip()
    return re.sub(r"([\\`*_#$~>\[\]])", r"\\\1", teks)


def buat_wordcloud(teks, stop, warna):
    wc = WordCloud(stopwords=stop, background_color="white", max_words=60,
                   width=800, height=500, random_state=42, collocations=False).generate(teks)
    cmap = plt.get_cmap(warna)

    def warna_kata(word, **kwargs):          # makin sering muncul, makin gelap
        r, g, b, _ = cmap(0.35 + 0.65 * wc.words_.get(word, 0) ** 0.5)
        return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

    return wc.recolor(color_func=warna_kata).to_array()


@st.cache_data
def hitung(path_csv, path_sw):
    df = pd.read_csv(path_csv, encoding="utf-8-sig")
    stop = set(Path(path_sw).read_text(encoding="utf-8").split())

    teks_asli = "full_text" if "full_text" in df.columns else "text"
    kurang = [k for k in (teks_asli, "clean_text", "cluster") if k not in df.columns]
    if kurang:
        raise ValueError("Kolom wajib tidak tersedia: " + ", ".join(kurang))

    n_awal = len(df)
    df = df.dropna(subset=["clean_text", "cluster"]).reset_index(drop=True)
    df["cluster"] = df["cluster"].astype(int)

    # TF-IDF: ubah teks menjadi angka. Kata umum (>40% postingan) dan kata langka (1 postingan) dibuang.
    vec = TfidfVectorizer(stop_words=list(stop), max_df=0.4, min_df=2)
    X = vec.fit_transform(df["clean_text"])
    kata = vec.get_feature_names_out()

    hasil = {}
    for i, k in enumerate(sorted(df["cluster"].unique())):
        idx = np.where(df["cluster"] == k)[0]
        pusat = np.asarray(X[idx].mean(axis=0)).ravel()            # rata-rata TF-IDF anggota cluster
        dominan = kata[pusat.argsort()[::-1][:5]]                   # 5 kata berbobot tertinggi
        dekat = np.asarray(X[idx].dot(pusat)).ravel().argsort()[::-1][:3]   # 3 postingan terdekat ke pusat
        teks = " ".join(df.loc[idx, "clean_text"])
        hasil[int(k)] = {
            "jumlah": len(idx),
            "dominan": list(dominan),
            "contoh": df.loc[idx[dekat], teks_asli].tolist(),
            "top": Counter(w for w in teks.split() if w not in stop and len(w) > 2).most_common(10),
            "wordcloud": buat_wordcloud(teks, stop, WARNA[i % len(WARNA)]),
        }
    return hasil, len(df), n_awal - len(df)


# ------------------------------- Halaman -------------------------------
st.title("🧩 Clustering Teks")
st.caption("TF-IDF + K-Means, 3 cluster. Label cluster berasal dari notebook.")

path_data, path_sw = cari_berkas(DATASET_FILE), cari_berkas(STOPWORDS_FILE)
if not (path_data and path_sw):
    st.error(f"Berkas '{DATASET_FILE}' atau '{STOPWORDS_FILE}' tidak ditemukan.")
    st.stop()

try:
    hasil, total, dibuang = hitung(str(path_data), str(path_sw))
except Exception as e:
    st.error(f"Data tidak dapat diproses: {e}")
    st.stop()

if dibuang:
    st.info(f"{dibuang} baris dikeluarkan karena teks bersih atau label cluster kosong.")

for k, tab in zip(hasil, st.tabs([f"Cluster {k}" for k in hasil])):
    h = hasil[k]
    with tab:
        st.subheader(TEMA.get(k, f"Cluster {k}"))
        st.metric("Jumlah postingan", f"{h['jumlah']} ({h['jumlah'] / total:.1%})")

        st.markdown("**Lima kata dominan:** " + ", ".join(f"`{w}`" for w in h["dominan"]))

        st.markdown("**Tiga contoh postingan:**")
        for n, teks in enumerate(h["contoh"], 1):
            st.markdown(f"{n}. {aman(teks)}")

        kata_top, jumlah_top = zip(*h["top"])
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(kata_top[::-1], jumlah_top[::-1])
        ax.bar_label(ax.containers[0])
        ax.set_title("Word count: 10 kata paling sering muncul")
        st.pyplot(fig)
        plt.close(fig)

        st.image(h["wordcloud"], caption="Word cloud")