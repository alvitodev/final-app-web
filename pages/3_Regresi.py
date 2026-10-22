import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATASET = "krl_clean.csv"
MODEL = "model_regresi_likes.pkl"


def load_model(model_file):
    return joblib.load(model_file)


def load_data(csv_file):
    df = pd.read_csv(csv_file).rename(columns={"full_text": "text", "favorite_count": "like_count"})
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce")
    df = df.dropna(subset=["text", "like_count"])      # likes kosong dibuang, tidak diganti 0

    # Fitur dihitung dari teks asli
    df["jumlah_kata"] = df["text"].apply(lambda t: len(re.sub(r"https?://\S+|www\.\S+", " ", str(t)).split()))
    df["jumlah_hashtag"] = df["text"].apply(lambda t: len(re.findall(r"#\w+", str(t))))
    return df


def main():
    st.title("📈 Regresi Linear: Prediksi Jumlah Likes")

    try:
        df = load_data(DATASET)
        model = load_model(MODEL)
    except FileNotFoundError:
        st.error("File dataset atau model tidak ditemukan. Pastikan berada di folder yang sama dengan aplikasi.")
        return
    except Exception as e:
        st.error(f"Data atau model tidak dapat dimuat: {e}")
        return

    fitur = ["jumlah_kata", "jumlah_hashtag"]
    X = df[fitur].to_numpy()
    y = df["like_count"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # 80% latih, 20% uji

    # 1. Korelasi
    st.subheader("1. Korelasi antar variabel")
    korelasi = df[fitur + ["like_count"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    gambar = ax.imshow(korelasi, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(korelasi.columns, rotation=30)
    ax.set_yticks(range(3))
    ax.set_yticklabels(korelasi.columns)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{korelasi.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(gambar)
    st.pyplot(fig)

    # 2. Evaluasi pada data uji
    y_pred = np.ravel(model.predict(X_test))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("2. Evaluasi pada data uji")
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:,.1f}")
    col2.metric("R²", f"{r2:.3f}")

    # 3. Grafik aktual vs prediksi
    st.subheader("3. Likes aktual vs prediksi")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(y_test, y_pred, alpha=0.7)
    batas = max(y_test.max(), y_pred.max())
    ax.plot([0, batas], [0, batas], "r--", label="Prediksi sempurna")
    ax.set_xlabel("Likes aktual")
    ax.set_ylabel("Likes prediksi")
    ax.legend()
    st.pyplot(fig)

    # 4. Formulir estimasi
    st.subheader("4. Estimasi likes")
    kata = st.number_input("Jumlah kata:", min_value=1, max_value=300, value=30)
    hashtag = st.number_input("Jumlah hashtag:", min_value=0, max_value=30, value=0)

    if st.button("Estimasi"):
        estimasi = float(np.ravel(model.predict([[kata, hashtag]]))[0])
        st.success("Proses estimasi selesai!")
        st.info(f"Estimasi jumlah likes: {estimasi:,.1f}")
        if estimasi < 0:
            st.warning("Estimasi negatif tidak bermakna sebagai jumlah likes dan menunjukkan keterbatasan model.")

    # 5. Interpretasi
    st.subheader("5. Interpretasi")
    st.markdown(f"""
- **MAE = {mae:,.1f}.** Rata-rata selisih antara likes aktual dan prediksi, dalam satuan likes.
  Bandingkan dengan median likes ({np.median(y):,.0f}) dan rata-rata likes ({y.mean():,.0f}).
- **R² = {r2:.3f}.** Porsi variasi likes yang dijelaskan model. 1 = sempurna, 0 = sama dengan menebak rata-rata,
  negatif = lebih buruk daripada menebak rata-rata.
- **Keterbatasan dua fitur:** likes juga dipengaruhi usia postingan, jangkauan akun, dan isi pesan yang tidak ada di model.
  Likes sangat timpang sehingga beberapa postingan viral mendominasi hasil. Regresi linear bisa menghasilkan
  estimasi negatif ({int((y_pred < 0).sum())} dari {len(y_pred)} prediksi data uji), yang tidak bermakna sebagai jumlah likes.
""")


if __name__ == "__main__":
    main()