
import streamlit as st
from common import get_data, prepare_data

st.set_page_config(
    page_title="Analisis KRL",
    page_icon="🚆",
    layout="wide",
)

st.title("🚆 Dashboard Analisis Percakapan KRL")
st.write(
    "Aplikasi untuk mengeksplorasi data percakapan terkait "
    "KRL melalui analisis sentimen, clustering, dan regresi."
)

df = get_data()

if df.empty:
    st.warning(
        "Dataset belum ditemukan. Buka halaman Data untuk mengunggah CSV."
    )
    st.stop()

data = prepare_data(df)

c1, c2, c3 = st.columns(3)
c1.metric("Total postingan", f"{len(data):,}")
c2.metric("Total likes", f"{data['_likes'].fillna(0).sum():,.0f}")
c3.metric("Rata-rata kata", f"{data['_word_count'].mean():.1f}")

st.subheader("Pratinjau dataset")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Temuan awal")
st.markdown(
    """
    - Jumlah postingan dan interaksi dapat digunakan untuk melihat
      tingkat percakapan tentang KRL.
    - Pengelompokan teks dapat membantu menemukan tema pembicaraan.
    - Hubungan antara karakteristik postingan dan likes tidak otomatis
      membuktikan hubungan sebab-akibat.
    """
)

st.subheader("Keterbatasan")
st.markdown(
    """
    1. Label sentimen berbasis kamus sederhana dapat salah memahami
       sarkasme, negasi, singkatan, dan konteks bahasa Indonesia.
    2. Data media sosial tidak selalu mewakili seluruh pengguna KRL;
       data yang tersedia juga dapat mengandung nilai kosong atau bias.
    """
)

st.info(
    "Pilih Data, Sentimen, Clustering, atau Regresi pada navigasi "
    "sidebar untuk membuka analisis."
)
