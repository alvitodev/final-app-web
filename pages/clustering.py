
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from common import get_data, prepare_data

st.title("🧩 Analisis Clustering")

df = get_data()
if df.empty:
    st.warning("Unggah dataset melalui halaman Data terlebih dahulu.")
    st.stop()

data = prepare_data(df)
text_col = "clean_text" if "clean_text" in df.columns else (
    "text_clean" if "text_clean" in df.columns else (
        "full_text" if "full_text" in df.columns else "text"
    )
)

if text_col not in df.columns:
    st.error("Kolom teks tidak ditemukan.")
    st.stop()

data["_cluster_text"] = df[text_col].fillna("").astype(str)
data = data[data["_cluster_text"].str.strip().ne("")].copy()

if data.empty:
    st.warning("Tidak ada teks yang dapat dianalisis.")
    st.stop()

if "cluster" in df.columns:
    data["Cluster"] = df.loc[data.index, "cluster"].astype(str)
    st.info("Menggunakan label cluster yang sudah ada di dataset.")
else:
    n_clusters = st.slider(
        "Jumlah cluster",
        min_value=2,
        max_value=8,
        value=4,
    )

    try:
        vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words=None,
        )
        X = vectorizer.fit_transform(data["_cluster_text"])

        if X.shape[1] == 0 or X.shape[0] < n_clusters:
            st.error("Teks terlalu sedikit untuk jumlah cluster tersebut.")
            st.stop()

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        data["Cluster"] = model.fit_predict(X).astype(str)

        terms = vectorizer.get_feature_names_out()
        st.subheader("Kata dominan per cluster")

        for i, center in enumerate(model.cluster_centers_):
            top_idx = center.argsort()[::-1][:8]
            st.write(
                f"Cluster {i}: "
                + ", ".join(terms[j] for j in top_idx)
            )
    except ValueError as exc:
        st.error(f"Clustering gagal: {exc}")
        st.stop()

summary = (
    data.groupby("Cluster")
    .agg(
        Ukuran=("_cluster_text", "size"),
        Contoh_postingan=("_cluster_text", "first"),
    )
    .reset_index()
)

if "tema_cluster" in df.columns:
    themes = df.loc[data.index, ["tema_cluster"]].copy()
    themes["Cluster"] = data["Cluster"].values
    themes = (
        themes.groupby("Cluster")["tema_cluster"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "-")
        .rename("Nama tema")
        .reset_index()
    )
    summary = summary.merge(themes, on="Cluster", how="left")
else:
    summary["Nama tema"] = summary["Cluster"].apply(
        lambda x: f"Tema {x}"
    )

st.subheader("Ukuran cluster dan contoh postingan")
st.dataframe(summary, use_container_width=True)

st.subheader("Distribusi ukuran cluster")
fig, ax = plt.subplots()
summary.plot(
    x="Cluster",
    y="Ukuran",
    kind="bar",
    ax=ax,
    legend=False,
)
ax.set_xlabel("Cluster")
ax.set_ylabel("Jumlah postingan")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Catatan: kata dominan membantu menginterpretasikan tema. "
    "Nama tema perlu ditinjau secara manual agar sesuai isi postingan."
)
