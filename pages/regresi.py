
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from common import get_data, prepare_data

st.title("📈 Analisis Regresi")

df = get_data()
if df.empty:
    st.warning("Unggah dataset melalui halaman Data terlebih dahulu.")
    st.stop()

data = prepare_data(df)

# Pilih fitur yang tersedia pada dataset.
features = ["_word_count", "_hashtags"]
source_features = {
    "_word_count": "jumlah_kata",
    "_hashtags": "jumlah_hashtag",
    "reply_count": "reply_count",
    "retweet_count": "retweet_count",
    "quote_count": "quote_count",
}

for feature, source in source_features.items():
    if source in df.columns and feature not in features:
        features.append(feature)

for feature in features:
    if feature not in data.columns:
        data[feature] = pd.to_numeric(
            df[source_features.get(feature, feature)],
            errors="coerce",
        ) if source_features.get(feature, feature) in df.columns else 0

data["_likes"] = pd.to_numeric(data["_likes"], errors="coerce")
model_data = data[features + ["_likes"]].replace(
    [np.inf, -np.inf], np.nan
).dropna()

model_data = model_data[model_data["_likes"] >= 0]

st.write("Fitur prediksi:", ", ".join(features))
st.metric("Baris yang digunakan", f"{len(model_data):,}")

if len(model_data) < 10:
    st.warning(
        "Dibutuhkan minimal 10 baris data valid untuk menjalankan "
        "contoh regresi ini."
    )
    st.stop()

X = model_data[features]
y = model_data["_likes"]

if y.nunique() < 2:
    st.error("Target likes harus memiliki variasi nilai.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred) if len(y_test) >= 2 else float("nan")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "Tidak tersedia")

st.subheader("Grafik aktual vs prediksi")
fig, ax = plt.subplots()
ax.scatter(y_test, pred)
ax.set_xlabel("Likes aktual")
ax.set_ylabel("Likes prediksi")
ax.set_title("Aktual vs Prediksi")
st.pyplot(fig)
plt.close(fig)

comparison = pd.DataFrame({
    "Likes aktual": y_test.to_numpy(),
    "Likes prediksi": np.round(pred, 2),
})
st.dataframe(comparison.head(50), use_container_width=True)

st.subheader("Estimasi likes")
st.caption(
    "Estimasi berikut bersifat perkiraan berdasarkan model yang "
    "dilatih pada dataset, bukan jaminan jumlah likes yang akan didapat."
)

values = {}
for feature in features:
    default = float(model_data[feature].median())
    values[feature] = st.number_input(
        feature,
        min_value=0.0,
        value=max(0.0, default),
        step=1.0,
    )

if st.button("Estimasi jumlah likes"):
    new_data = pd.DataFrame([values], columns=features)
    estimate = max(0.0, float(model.predict(new_data)[0]))
    st.success(f"Estimasi jumlah likes: {estimate:.0f}")

st.warning(
    "Keterbatasan: performa regresi dipengaruhi ukuran dan kualitas "
    "dataset. Nilai R² negatif dapat terjadi jika model kurang baik."
)
