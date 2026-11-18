import os
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from config import FEATURE_COLS, REGRESSION_TARGETS
from preprocess import fetch_raw_data, clean_dataframe, resample_hourly_and_features, build_supervised_dataset


def train_and_save_models():
    raw_df = fetch_raw_data()
    clean_df = clean_dataframe(raw_df)
    hourly_df = resample_hourly_and_features(clean_df)
    df = build_supervised_dataset(hourly_df)

    if len(df) < 120 or df["target_class"].nunique() < 2:
        print(
            f"[SKIP] Data asli baru terkumpul {len(df)} jam ({df['target_class'].nunique()} kelas). "
            "Pelatihan ulang ditunda hingga minimal 120 jam (5 hari) dan >=2 kelas cuaca terkumpul."
        )
        return

    # Temporal Split (80% data awal = Latih, 20% data akhir = Uji)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLS]
    y_train_cls = train_df["target_class"]

    X_test = test_df[FEATURE_COLS]
    y_test_cls = test_df["target_class"]

    print(f"[INFO] Pembagian Temporal -> Data Latih: {len(X_train)} | Data Uji: {len(X_test)}")

    standalone_scaler = StandardScaler()
    standalone_scaler.fit(X_train)

    # Pelatihan Support Vector Classifier (SVC) - Two-Stage Fitting
    min_class_count = y_train_cls.value_counts().min()
    n_splits = max(2, min(5, int(min_class_count)))

    pipe_svc = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=False, random_state=42)),
    ])

    param_grid_svc = {
        "svc__kernel": ["rbf", "linear"],
        "svc__C": [1.0, 10.0, 100.0],
        "svc__gamma": ["scale", 0.01],
        "svc__class_weight": ["balanced", None],
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid_svc = GridSearchCV(pipe_svc, param_grid_svc, cv=cv, scoring="f1_weighted", n_jobs=-1)
    grid_svc.fit(X_train, y_train_cls)

    best_params = grid_svc.best_params_
    print(f"[INFO] Parameter terbaik SVC: {best_params}")

    final_svc = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=best_params["svc__kernel"],
            C=best_params["svc__C"],
            gamma=best_params["svc__gamma"],
            class_weight=best_params["svc__class_weight"],
            probability=True,
            random_state=42,
        )),
    ])
    final_svc.fit(X_train, y_train_cls)

    y_pred_cls = final_svc.predict(X_test)
    svc_acc = accuracy_score(y_test_cls, y_pred_cls)
    svc_f1 = f1_score(y_test_cls, y_pred_cls, average="weighted")
    print(f"[METRIK SVC] Accuracy: {svc_acc:.4f} | Weighted F1: {svc_f1:.4f}")

    # Pelatihan 4 Estimator Support Vector Regressor (SVR) Independen
    svr_models = {}
    svr_metrics = {}
    train_bounds = {}

    for target in REGRESSION_TARGETS:
        y_tr = train_df[f"target_{target}"]
        y_te = test_df[f"target_{target}"]

        train_bounds[target] = {
            "min": float(y_tr.min()),
            "max": float(y_tr.max()),
        }

        if target == "hujan":
            pipe_svr = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1)),
            ])
        elif target == "kecepatan_angin":
            pipe_svr = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="linear", C=1.0, epsilon=0.1)),
            ])
        else:
            pipe_svr = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="linear", C=100.0, epsilon=0.1)),
            ])

        pipe_svr.fit(X_train, y_tr)
        preds = pipe_svr.predict(X_test)

        mae = mean_absolute_error(y_te, preds)
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        svr_models[target] = pipe_svr
        svr_metrics[target] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}
        print(f"[METRIK SVR - {target}] MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_svc, "models/svm_classifier.pkl")
    joblib.dump(svr_models, "models/svm_regressor.pkl")
    joblib.dump(standalone_scaler, "models/scaler.pkl")

    meta = {
        "trained_at_wib": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": FEATURE_COLS,
        "classes": list(final_svc.classes_),
        "svc_best_params": best_params,
        "svc_metrics": {"accuracy": round(svc_acc, 4), "f1_weighted": round(svc_f1, 4)},
        "svr_metrics": svr_metrics,
        "train_bounds": train_bounds,
    }

    with open("models/model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Seluruh artefak model berhasil disimpan di folder models/.")


if __name__ == "__main__":
    train_and_save_models()
