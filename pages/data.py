import streamlit as st
import pandas as pd
from pathlib import Path

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Analisis KRL - Data",
    page_icon="📂",
    layout="wide"
)

# ==================================================
# MEMUAT DATASET OTOMATIS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "krl_clean.csv"


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path, low_memory=False)


st.title("📂 Data")

# ==================================================
# VALIDASI FILE DATASET
# ==================================================
if not DATASET_PATH.exists():
    st.error(
        f"File dataset tidak ditemukan: {DATASET_PATH}"
    )
    st.info(
        "Pastikan krl_clean.csv berada satu folder dengan app.py."
    )
    st.stop()

# ==================================================
# MEMBACA DATASET
# ==================================================
try:
    df = load_dataset(str(DATASET_PATH))

    # Membersihkan spasi pada nama kolom
    df.columns = df.columns.astype(str).str.strip()

except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

if df.empty:
    st.warning("Dataset tidak memiliki baris data.")
    st.stop()

st.success("Dataset krl_clean.csv berhasil dimuat")

# ==================================================
# RINGKASAN DATASET
# ==================================================
st.subheader("📊 Ringkasan Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Baris", f"{len(df):,}")

with col2:
    st.metric("Jumlah Kolom", f"{len(df.columns):,}")

with col3:
    st.metric(
        "Total Sel Kosong",
        f"{int(df.isna().sum().sum()):,}"
    )

# ==================================================
# PEMERIKSAAN KOLOM
# ==================================================
st.subheader("🔎 Pemeriksaan Kolom")

required_columns = {
    "full_text": "Teks postingan asli",
    "clean_text": "Teks yang sudah dibersihkan",
    "created_at": "Tanggal postingan",
    "favorite_count": "Jumlah likes",
    "cluster": "Label cluster",
    "tema_cluster": "Tema cluster"
}

for column, description in required_columns.items():
    if column in df.columns:
        st.success(f"{description}: tersedia (`{column}`)")
    else:
        st.warning(f"{description}: tidak tersedia (`{column}`)")

# ==================================================
# DAFTAR KOLOM
# ==================================================
st.subheader("📋 Daftar Kolom")
st.write(", ".join(df.columns.tolist()))

# ==================================================
# PRATINJAU DATASET
# ==================================================
st.subheader("🗃️ Pratinjau Dataset")

jumlah_maksimum = min(100, len(df))
opsi_baris = sorted(
    set([5, 10, 20, 50, jumlah_maksimum])
)

opsi_baris = [
    n for n in opsi_baris
    if n <= jumlah_maksimum and n > 0
]

jumlah_baris = st.selectbox(
    "Jumlah baris yang ditampilkan",
    options=opsi_baris,
    index=opsi_baris.index(min(10, jumlah_maksimum))
)

st.dataframe(
    df.head(jumlah_baris),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Menampilkan {min(jumlah_baris, len(df)):,} "
    f"dari {len(df):,} baris."
)

# ==================================================
# JUMLAH DATA KOSONG PER KOLOM
# ==================================================
st.subheader("🧹 Jumlah Data Kosong per Kolom")

missing_df = pd.DataFrame({
    "Nama Kolom": df.columns,
    "Jumlah Kosong": df.isna().sum().values,
    "Persentase Kosong (%)": (
        df.isna().mean().values * 100
    ).round(2)
})

st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True
)

# ==================================================
# INFORMASI DATASET
# ==================================================
with st.expander("ℹ️ Informasi Dataset"):
    st.write(f"**Nama file:** {DATASET_PATH.name}")
    st.write(f"**Lokasi file:** {DATASET_PATH}")
    st.write(f"**Jumlah baris:** {len(df):,}")
    st.write(f"**Jumlah kolom:** {len(df.columns):,}")
    st.write(f"**Total duplikat:** {int(df.duplicated().sum()):,}")

# ==================================================
# UNDUH DATASET
# ==================================================
st.subheader("⬇️ Unduh Dataset")

csv_data = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Unduh krl_clean.csv",
    data=csv_data,
    file_name="krl_clean.csv",
    mime="text/csv"
)