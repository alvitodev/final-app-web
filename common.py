
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

DATASETS = [
    "dataset_krl_bersih.csv",
    "krl_clean.csv",
    "krl.csv",
]


def load_default_data():
    for filename in DATASETS:
        path = BASE_DIR / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                return df, filename
            except Exception as exc:
                st.error(f"Gagal membaca {filename}: {exc}")

    return pd.DataFrame(), None


def get_data():
    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"].copy()
        df.columns = df.columns.str.strip()
        return df

    df, _ = load_default_data()
    return df


def get_text_column(df):
    for col in ["text_clean", "clean_text", "full_text", "text"]:
        if col in df.columns:
            return col
    return None


def get_like_column(df):
    for col in ["like_count", "favorite_count", "likes"]:
        if col in df.columns:
            return col
    return None


def sentiment_label(text):
    positive = {
        "bagus", "baik", "aman", "nyaman", "cepat", "terima kasih",
        "terimakasih", "membantu", "apresiasi", "hebat", "lancar",
        "puas", "teratur", "solidaritas", "sigap", "terbaik",
    }
    negative = {
        "buruk", "kecewa", "lambat", "terlambat", "macet", "parah",
        "marah", "kecewa", "gangguan", "kecelakaan", "korban",
        "bahaya", "rusak", "sesak", "panik", "sedih", "meninggal",
    }

    words = str(text).lower().split()
    score = sum(word.strip(".,!?;:#") in positive for word in words)
    score -= sum(word.strip(".,!?;:#") in negative for word in words)

    if score > 0:
        return "Positif"
    if score < 0:
        return "Negatif"
    return "Netral"


def prepare_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    text_col = get_text_column(df)
    like_col = get_like_column(df)

    if text_col:
        df["_text"] = df[text_col].fillna("").astype(str)
    else:
        df["_text"] = ""

    if like_col:
        df["_likes"] = pd.to_numeric(df[like_col], errors="coerce")
    else:
        df["_likes"] = 0

    if "created_at" in df.columns:
        df["_date"] = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True
        )
    else:
        df["_date"] = pd.NaT

    df["_word_count"] = df["_text"].str.split().str.len()

    if "jumlah_hashtag" in df.columns:
        df["_hashtags"] = pd.to_numeric(
            df["jumlah_hashtag"], errors="coerce"
        ).fillna(0)
    else:
        df["_hashtags"] = df["_text"].str.count(r"#\w+")

    return df