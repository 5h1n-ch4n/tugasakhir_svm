import os
import json
import numpy as np
import pandas as pd

try:
    import firebase_admin
    from firebase_admin import credentials, db
    HAS_FIREBASE = True
except ImportError:
    firebase_admin = None
    credentials = None
    db = None
    HAS_FIREBASE = False

from config import (
    FIREBASE_HISTORY_PATH,
    PHYSICAL_LIMITS,
    FEATURE_COLS,
    REGRESSION_TARGETS,
    label_from_features,
)


def init_firebase() -> bool:
    """Inisialisasi Firebase Admin SDK dari GitHub Secrets atau file lokal."""
    if not HAS_FIREBASE:
        return False

    if firebase_admin._apps:
        return True

    db_url = os.environ.get("FIREBASE_DB_URL")
    cred_json = os.environ.get("FIREBASE_CREDENTIALS")

    if not db_url:
        return False

    if cred_json:
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
    elif os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
    else:
        return False

    firebase_admin.initialize_app(cred, {"databaseURL": db_url})
    return True


def fetch_raw_data(limit: int = 50000, local_csv: str = "data/raw_history.csv") -> pd.DataFrame:
    """Mengambil data dari Firebase Realtime Database atau fallback ke CSV lokal."""
    if init_firebase():
        ref = db.reference(FIREBASE_HISTORY_PATH)
        raw_dict = ref.order_by_key().limit_to_last(limit).get()
        if raw_dict:
            records = []
            for key, val in raw_dict.items():
                if isinstance(val, dict):
                    cuaca = val.get("cuaca", val)
                    records.append({
                        "timestamp": val.get("timestamp", key),
                        "suhu": cuaca.get("suhu", cuaca.get("suhu_c")),
                        "kelembapan": cuaca.get("kelembapan", cuaca.get("kelembapan_persen")),
                        "tekanan": cuaca.get("tekanan", cuaca.get("tekanan_hpa")),
                        "cahaya": cuaca.get("cahaya", cuaca.get("cahaya_lux")),
                        "hujan": cuaca.get("hujan", cuaca.get("hujan_interval_mm", 0.0)),
                        "kecepatan_angin": cuaca.get("kecepatan_angin", cuaca.get("angin_kmh", 0.0)),
                    })
            df = pd.DataFrame(records)
            if not df.empty:
                os.makedirs(os.path.dirname(local_csv), exist_ok=True)
                df.to_csv(local_csv, index=False)
                print(f"[INFO] Mengambil {len(df)} rekaman dari Firebase.")
                return df

    if os.path.exists(local_csv):
        print(f"[INFO] Membaca dataset lokal dari {local_csv}.")
        return pd.read_csv(local_csv)

    raise FileNotFoundError("Data Firebase kosong dan file data/raw_history.csv tidak ditemukan.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validasi numerik, zona waktu WIB, deduplikasi, dan filtering outlier fisis."""
    df = df.copy()

    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_num.notna().sum() > 0:
        unit = "ms" if ts_num.median() > 1e11 else "s"
        df["datetime"] = pd.to_datetime(ts_num, unit=unit, utc=True).dt.tz_convert("Asia/Jakarta")
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(
            "Asia/Jakarta", ambiguous="NaT", nonexistent="NaT"
        )

    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="last")

    for col, (low, high) in PHYSICAL_LIMITS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].between(low, high)]

    return df.set_index("datetime")


def resample_hourly_and_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Agregasi per jam (1h) dan pembentukan 12 fitur kausal."""
    hourly = df_clean.resample("1h").agg({
        "suhu": "mean",
        "kelembapan": "mean",
        "tekanan": "mean",
        "cahaya": "max",
        "hujan": "sum",
        "kecepatan_angin": "mean",
    })

    hourly = hourly.dropna(subset=["suhu", "kelembapan"]).copy()

    hourly["temp_change_1h"] = hourly["suhu"].diff(1)
    hourly["humidity_change_1h"] = hourly["kelembapan"].diff(1)
    hourly["pressure_change_1h"] = hourly["tekanan"].diff(1)

    hourly["rain_sum_3h"] = hourly["hujan"].rolling(window=3, min_periods=1).sum()
    hourly["temp_avg_3h"] = hourly["suhu"].rolling(window=3, min_periods=1).mean()
    hourly["humidity_avg_3h"] = hourly["kelembapan"].rolling(window=3, min_periods=1).mean()

    hourly = hourly.dropna(subset=FEATURE_COLS).copy()

    hourly["weather_class"] = hourly.apply(
        lambda r: label_from_features(r["hujan"], r["kelembapan"], r["cahaya"]), axis=1
    )
    return hourly


def build_supervised_dataset(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Menyusun pasangan X(t) -> y(t+1) dengan verifikasi selisih waktu tepat 1 jam."""
    df = hourly_df.reset_index().copy()

    df["target_datetime"] = df["datetime"].shift(-1)
    df["target_class"] = df["weather_class"].shift(-1)
    for col in REGRESSION_TARGETS:
        df[f"target_{col}"] = df[col].shift(-1)

    delta_hours = (df["target_datetime"] - df["datetime"]).dt.total_seconds() / 3600.0
    valid_mask = delta_hours.between(0.95, 1.05)

    supervised = df[valid_mask].dropna().reset_index(drop=True)
    return supervised


if __name__ == "__main__":
    raw_df = fetch_raw_data()
    clean_df = clean_dataframe(raw_df)
    hourly_df = resample_hourly_and_features(clean_df)
    supervised_df = build_supervised_dataset(hourly_df)

    os.makedirs("data", exist_ok=True)
    supervised_df.to_csv("data/dataset_aws.csv", index=False)
    print(f"[OK] Dataset bersih siap latih: {len(supervised_df)} sampel per jam disimpan ke data/dataset_aws.csv")
