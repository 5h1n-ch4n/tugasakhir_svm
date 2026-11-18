import os
import json
import joblib
import numpy as np
import pandas as pd

try:
    from firebase_admin import db
except ImportError:
    db = None

from config import (
    FIREBASE_PREDICTION_PATH,
    FEATURE_COLS,
    REGRESSION_TARGETS,
    build_recommendation,
)
from preprocess import init_firebase, fetch_raw_data, clean_dataframe, resample_hourly_and_features


def run_hourly_prediction():
    svc_model = joblib.load("models/svm_classifier.pkl")
    svr_models = joblib.load("models/svm_regressor.pkl")
    with open("models/model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw_df = fetch_raw_data(limit=300)
    clean_df = clean_dataframe(raw_df)
    hourly_df = resample_hourly_and_features(clean_df)

    if hourly_df.empty:
        raise RuntimeError("Data per jam tidak mencukupi untuk menyusun konteks 3 jam.")

    latest_row = hourly_df.iloc[[-1]].copy()
    current_time = latest_row.index[0]
    target_time = current_time + pd.Timedelta(hours=1)

    X_latest = latest_row[FEATURE_COLS].copy()
    if X_latest.isna().any().any():
        X_latest = X_latest.fillna(hourly_df[FEATURE_COLS].median())

    pred_class = str(svc_model.predict(X_latest)[0])
    proba_arr = svc_model.predict_proba(X_latest)[0]
    probabilities = {
        str(cls): round(float(prob) * 100.0, 2)
        for cls, prob in zip(svc_model.classes_, proba_arr)
    }

    pred_numeric = {}
    bounds = meta.get("train_bounds", {})
    for target in REGRESSION_TARGETS:
        val = float(svr_models[target].predict(X_latest)[0])
        if target in bounds:
            val = float(np.clip(val, bounds[target]["min"], bounds[target]["max"]))
        if target == "hujan" and val < 0.05:
            val = 0.0
        pred_numeric[target] = round(val, 2)

    # Susun rekomendasi khusus kandang ayam pejantan (fokus THI & kontrol kipas)
    rec_result = build_recommendation(pred_class, pred_numeric)

    payload = {
        "station_id": "AWS_TPST_01",
        "object_type": "Kandang Ayam Pejantan",
        "observation_time": current_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "forecast_time": target_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "predicted_class": pred_class,
        "probabilities_pct": probabilities,
        "forecast_values": pred_numeric,
        "thi_forecast": rec_result["thi_forecast"],
        "heat_stress_status": rec_result["heat_stress_status"],
        "fan_recommendation": rec_result["fan_recommendation"],
        "recommendations": rec_result["messages"],
    }

    os.makedirs("models", exist_ok=True)
    with open("models/latest_prediction.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if init_firebase() and db is not None:
        db.reference(FIREBASE_PREDICTION_PATH).set(payload)
        print(f"[FIREBASE OK] Prediksi untuk {payload['forecast_time']} berhasil ditulis.")
    else:
        print("[LOCAL OK] Prediksi selesai (mode offline, disimpan di models/latest_prediction.json):")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_hourly_prediction()
