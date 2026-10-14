import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATASET_FILE = "krl_clean.csv" 
MODEL_FILE = "model_regresi_likes.pkl"
RANDOM_STATE = 42       
UKURAN_UJI = 0.2           
MIN_DATA = 20
FITUR = ["jumlah_kata", "jumlah_hashtag"]
ALIAS_KOLOM = {"full_text": "text", "favorite_count": "like_count"}


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


@st.cache_data
def muat_dataset(path, waktu_ubah):
    # waktu_ubah ikut menjadi kunci cache, jadi berkas yang diganti otomatis dibaca ulang
    return pd.read_csv(path)


@st.cache_resource
def muat_model(path, waktu_ubah):
    return joblib.load(path)


def hitung_kata(teks):
    """Jumlah kata dari teks asli: token dipisah spasi, tanpa URL."""
    tanpa_url = re.sub(r"https?://\S+|www\.\S+", " ", str(teks))
    return len(tanpa_url.split())


def hitung_hashtag(teks):
    return len(re.findall(r"#\w+", str(teks)))


def siapkan_data(df):
    """Hitung fitur dari teks asli dan pilih baris dengan target valid.

    Likes yang kosong atau tidak valid TIDAK diganti nol.
    Mengembalikan (data, jumlah_baris_dikeluarkan_per_alasan).
    """
    df = df.rename(columns={k: v for k, v in ALIAS_KOLOM.items()
                            if k in df.columns and v not in df.columns})
    hilang = [k for k in ("text", "like_count") if k not in df.columns]
    if hilang:
        raise ValueError("Kolom wajib tidak tersedia: " + ", ".join(hilang)
                         + " (atau full_text dan favorite_count).")

    data = df[["text", "like_count"]].copy()
    teks_ok = data["text"].notna() & (data["text"].astype(str).str.strip() != "")
    likes = pd.to_numeric(data["like_count"], errors="coerce")
    likes_ok = likes.notna() & (likes >= 0) & (likes % 1 == 0)

    alasan = {
        "teks kosong": int((~teks_ok).sum()),
        "jumlah likes kosong atau tidak valid": int((teks_ok & ~likes_ok).sum()),
    }
    data["like_count"] = likes
    data = data[teks_ok & likes_ok].copy()
    data["like_count"] = data["like_count"].astype(int)
    data["jumlah_kata"] = data["text"].apply(hitung_kata)
    data["jumlah_hashtag"] = data["text"].apply(hitung_hashtag)
    return data, alasan


# ------------------------------- Halaman -------------------------------
st.title("📈 Analisis Regresi Linear")
st.caption("Fitur: jumlah kata dan jumlah hashtag (dihitung dari teks asli). "
           "Target: jumlah likes yang tercatat saat data dikumpulkan.")

# 1. Muat dataset langsung dari berkas
path_data = cari_berkas(DATASET_FILE)
if path_data is None:
    st.error(f"Dataset '{DATASET_FILE}' tidak ditemukan. Letakkan berkas itu di folder aplikasi "
             "atau di subfolder data/, atau sesuaikan DATASET_FILE di kode.")
    st.stop()

try:
    df = muat_dataset(str(path_data), path_data.stat().st_mtime)
    data, alasan = siapkan_data(df)
except Exception as e:
    st.error(f"Data tidak dapat diproses untuk regresi: {e}")
    st.stop()

# 2. Jelaskan baris yang dikeluarkan
n_keluar = sum(alasan.values())
if n_keluar == 0:
    st.info(f"Seluruh {len(df)} baris dataset valid dan dipakai pada regresi.")
else:
    rincian = "; ".join(f"{n} baris karena {a}" for a, n in alasan.items() if n > 0)
    st.info(f"{n_keluar} dari {len(df)} baris dikeluarkan dari regresi: {rincian}. "
            "Likes yang tidak diketahui tidak diganti nol.")

if len(data) < MIN_DATA:
    st.warning(f"Data valid hanya {len(data)} baris (minimal {MIN_DATA}). Regresi tidak dijalankan.")
    st.stop()
if data["like_count"].nunique() < 2:
    st.error("Target likes tidak memiliki variasi nilai, regresi tidak dapat dilakukan.")
    st.stop()

# 3. Bagi data 80:20 dengan random_state tetap
X = data[FITUR]
y = data["like_count"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=UKURAN_UJI, random_state=RANDOM_STATE
)

# Load model pkl
model_latih_ulang = LinearRegression().fit(X_train.to_numpy(), y_train.to_numpy())
model = model_latih_ulang
sumber_model = "dilatih ulang dari data latih"

path_model = cari_berkas(MODEL_FILE)
if path_model is None:
    st.warning(f"Berkas model '{MODEL_FILE}' tidak ditemukan, jadi model dilatih ulang dari data latih.")
else:
    try:
        model_pkl = muat_model(str(path_model), path_model.stat().st_mtime)
        cocok = (
            getattr(model_pkl, "n_features_in_", None) == len(FITUR)
            and np.allclose(np.ravel(model_pkl.coef_), np.ravel(model_latih_ulang.coef_), rtol=1e-6, atol=1e-6)
            and np.allclose(np.ravel(model_pkl.intercept_), np.ravel(model_latih_ulang.intercept_), rtol=1e-6, atol=1e-6)
        )
        if cocok:
            model = model_pkl
            sumber_model = f"dimuat dari {MODEL_FILE}"
            st.success(f"Model {sumber_model}. Koefisiennya cocok dengan model yang dilatih dari data latih "
                       "halaman ini, jadi evaluasi pada data uji valid.")
        else:
            st.warning(f"Model di {MODEL_FILE} tidak cocok dengan data latih halaman ini (kemungkinan dilatih "
                       "pada dataset atau pembagian data yang berbeda). Agar evaluasi tidak bocor, halaman ini "
                       "memakai model yang dilatih ulang dari data latih.")
    except Exception as e:
        st.warning(f"Model {MODEL_FILE} tidak dapat dibaca ({e}). Model dilatih ulang dari data latih.")


def prediksi(X_nilai):
    return np.ravel(model.predict(np.asarray(X_nilai, dtype=float)))


# Evaluasi pada data uji
y_pred = prediksi(X_test.to_numpy())
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae_rata2 = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
n_negatif = int((y_pred < 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE (data uji)", f"{mae:,.1f}")
c2.metric("R² (data uji)", f"{r2:.3f}")
c3.metric("Data latih", len(X_train))
c4.metric("Data uji", len(X_test))
st.caption(f"Pembagian 80:20 dengan random_state={RANDOM_STATE}; model hanya dilatih memakai data latih. "
           f"Pembanding: menebak rata-rata likes data latih untuk semua postingan menghasilkan MAE {mae_rata2:,.1f}.")

with st.expander("Koefisien model"):
    koef = pd.DataFrame({
        "Komponen": ["Intercept"] + FITUR,
        "Nilai": [float(np.ravel(model.intercept_)[0])] + [float(v) for v in np.ravel(model.coef_)],
    }).round(2)
    st.dataframe(koef, hide_index=True)

# Grafik likes aktual vs prediksi
st.subheader("Grafik likes aktual dibandingkan hasil prediksi")
skala_log = st.checkbox("Gunakan skala symlog agar postingan dengan likes kecil terlihat", value=False)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(y_test, y_pred, alpha=0.7, color="steelblue")
batas = max(float(y_test.max()), float(np.max(y_pred)))
ax.plot([0, batas], [0, batas], "r--", label="Prediksi sempurna")
ax.axhline(0, color="gray", lw=0.8)
if skala_log:
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
ax.set_xlabel("Likes aktual")
ax.set_ylabel("Likes prediksi")
ax.legend()
st.pyplot(fig)
plt.close(fig)

with st.expander("Tabel perbandingan (50 baris pertama data uji)"):
    st.dataframe(pd.DataFrame({
        "Likes aktual": y_test.to_numpy(),
        "Likes prediksi": np.round(y_pred, 1),
    }).head(50), hide_index=True)

# Input estimasi likes
st.subheader("Formulir estimasi likes")
st.caption("Estimasi bersifat perkiraan kasar dari model yang lemah, bukan jaminan jumlah likes.")
with st.form("form_estimasi_likes"):
    kata = st.number_input("Jumlah kata dalam postingan", min_value=1, max_value=300,
                           value=int(data["jumlah_kata"].median()), step=1)
    tag = st.number_input("Jumlah hashtag dalam postingan", min_value=0, max_value=30, value=0, step=1)
    kirim = st.form_submit_button("Estimasi jumlah likes")

if kirim:
    estimasi = float(prediksi([[kata, tag]])[0])
    st.metric("Estimasi jumlah likes", f"{estimasi:,.1f}")
    if estimasi < 0:
        st.warning("Estimasi bernilai negatif sehingga tidak bermakna sebagai jumlah likes. Ini menunjukkan "
                   "keterbatasan model regresi linear dengan dua fitur ini. Nilainya sengaja tidak dipotong menjadi 0.")

st.warning("Keterbatasan data: kesimpulan hanya berlaku untuk sampel postingan pada dataset ini dan tidak "
           "mewakili seluruh percakapan di X.")