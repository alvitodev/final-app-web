
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from common import get_data, prepare_data, sentiment_label

st.title("💬 Analisis Sentimen")

df = get_data()
if df.empty:
    st.warning("Unggah dataset melalui halaman Data terlebih dahulu.")
    st.stop()

data = prepare_data(df)
data["Sentimen"] = data["_text"].apply(sentiment_label)

counts = (
    data["Sentimen"]
    .value_counts()
    .reindex(["Positif", "Netral", "Negatif"], fill_value=0)
)
percentages = counts / max(len(data), 1) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Positif", f"{counts['Positif']:,}",
          f"{percentages['Positif']:.1f}%")
c2.metric("Netral", f"{counts['Netral']:,}",
          f"{percentages['Netral']:.1f}%")
c3.metric("Negatif", f"{counts['Negatif']:,}",
          f"{percentages['Negatif']:.1f}%")

st.subheader("Distribusi sentimen")

fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Kategori")
ax.set_ylabel("Jumlah postingan")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

result = pd.DataFrame({
    "Kategori": counts.index,
    "Jumlah": counts.values,
    "Persentase (%)": percentages.values.round(2),
})
st.subheader("Tabel hasil")
st.dataframe(result, use_container_width=True)

st.subheader("Pemeriksaan manual")
sample_size = min(20, len(data))
sample = data.sample(sample_size, random_state=42) if sample_size else data

review = sample[["_text", "Sentimen"]].rename(
    columns={"_text": "Teks"}
)
review["Hasil pemeriksaan"] = ""

st.caption(
    "Periksa apakah label otomatis sesuai konteks. Isi kolom hasil "
    "pemeriksaan untuk mencatat penilaian manual."
)
edited = st.data_editor(review, use_container_width=True, num_rows="fixed")

if st.button("Simpan ringkasan pemeriksaan"):
    reviewed = edited["Hasil pemeriksaan"].astype(str).str.strip()
    completed = int(reviewed.ne("").sum())
    st.success(
        f"{completed} dari {len(edited)} baris sudah diberi catatan "
        "pemeriksaan pada sesi ini."
    )

st.warning(
    "Keterbatasan: klasifikasi berbasis kamus sederhana belum "
    "memahami konteks, sarkasme, atau negasi secara andal."
)
