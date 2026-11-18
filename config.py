# Konfigurasi Sistem Prediksi Cuaca & Kontrol Kipas Kandang Ayam Pejantan
# Stasiun: AWS_TPST_01 (TPST Universitas Diponegoro)

FIREBASE_HISTORY_PATH = "aws_data/AWS_TPST_01/history"
FIREBASE_PREDICTION_PATH = "predictions/AWS_TPST_01"

# Batas kelayakan fisis sensor iklim mikro tropis (Outlier Filtering)
PHYSICAL_LIMITS = {
    "suhu": (-10.0, 60.0),            # °C
    "kelembapan": (0.0, 100.0),       # % RH
    "tekanan": (800.0, 1100.0),       # hPa
    "cahaya": (0.0, 200000.0),        # lux
    "hujan": (0.0, 100.0),            # mm
    "kecepatan_angin": (0.0, 60.0),   # km/jam
}

# 12 Fitur Masukan Model (6 Dasar + 6 Turunan Temporal Kausal)
FEATURE_COLS = [
    "suhu",
    "kelembapan",
    "tekanan",
    "cahaya",
    "hujan",
    "kecepatan_angin",
    "temp_change_1h",
    "humidity_change_1h",
    "pressure_change_1h",
    "rain_sum_3h",
    "temp_avg_3h",
    "humidity_avg_3h",
]

REGRESSION_TARGETS = ["suhu", "kelembapan", "kecepatan_angin", "hujan"]

# Ambang Batas Stres Panas (Heat Stress) Ayam Pejantan
SUHU_PANAS_WASPADA = 29.5   # °C -> Mulai gerah, nafsu makan mulai turun
SUHU_PANAS_BAHAYA = 31.0    # °C -> Ayam malas makan (heat stress), wajib nyalakan kipas penuh
THI_WASPADA = 78.0          # Indeks THI mulai masuk zona waspada stres panas
THI_BAHAYA = 82.0           # Indeks THI zona bahaya stres panas unggas


def calculate_thi(temp_c: float, rh_pct: float) -> float:
    """
    Menghitung Temperature-Humidity Index (THI) untuk ternak unggas.
    Formula standar NRC / Thom:
    THI = (1.8 * T + 32) - [(0.55 - 0.0055 * RH) * (1.8 * T - 26)]
    """
    t_f = 1.8 * temp_c + 32.0
    thi = t_f - ((0.55 - 0.0055 * rh_pct) * (1.8 * temp_c - 26.0))
    return round(float(thi), 2)


def label_from_features(hujan: float, kelembapan: float, cahaya: float) -> str:
    """Aturan deterministik pseudo-labeling 5 kelas cuaca (untuk informasi cuaca)."""
    if hujan >= 5.0:
        return "Hujan Lebat"
    elif 0.2 <= hujan < 5.0 and kelembapan >= 86.0:
        return "Berang-berang"
    elif 0.2 <= hujan < 5.0 and kelembapan < 86.0:
        return "Hujan Ringan"
    elif hujan < 0.2 and (kelembapan >= 76.0 or cahaya <= 2000.0):
        return "Berawan"
    else:
        return "Cerah"


def build_recommendation(pred_class: str, pred_num: dict) -> dict:
    """
    Menyusun status kenyamanan & rekomendasi operasional khusus Ayam Pejantan:
    - Ayam pejantan tahan suhu dingin (tidak perlu pemanas saat dingin).
    - Fokus utama adalah mencegah stres panas (THI & suhu tinggi) agar nafsu makan tetap terjaga.
    - Prediksi hujan hanya ditampilkan sebagai informasi cuaca.
    """
    suhu_pred = float(pred_num.get("suhu", 28.0))
    rh_pred = float(pred_num.get("kelembapan", 70.0))
    hujan_pred = float(pred_num.get("hujan", 0.0))

    thi_pred = calculate_thi(suhu_pred, rh_pred)
    recs = []

    # 1. Informasi Cuaca & Hujan (Informatif saja)
    if pred_class in ["Hujan Lebat", "Hujan Ringan", "Berang-berang"]:
        weather_info = (
            f"Info Cuaca: Diprediksi {pred_class} (curah hujan ~{hujan_pred:.1f} mm). "
            "Ayam pejantan tahan suhu dingin, cukup pastikan ventilasi tetap lancar."
        )
    else:
        weather_info = f"Info Cuaca: Diprediksi {pred_class}."
    recs.append(weather_info)

    # 2. Evaluasi Stres Panas (THI & Suhu) -> Kontrol Kipas Kandang
    if suhu_pred >= SUHU_PANAS_BAHAYA or thi_pred >= THI_BAHAYA:
        heat_status = "BAHAYA STRES PANAS"
        fan_action = "NYALAKAN KIPAS PENUH (100%)"
        recs.append(
            f"[PERINGATAN PANAS] Suhu prediksi {suhu_pred:.1f}°C (THI: {thi_pred}). "
            "Ayam pejantan berisiko stres panas dan mogok makan! "
            "Segera nyalakan seluruh kipas exhaust/blower agar kandang adem dan pastikan air minum lancar."
        )
    elif suhu_pred >= SUHU_PANAS_WASPADA or thi_pred >= THI_WASPADA:
        heat_status = "WASPADA PANAS"
        fan_action = "NYALAKAN KIPAS SEDANG (50-75%)"
        recs.append(
            f"[WASPADA GERAH] Suhu prediksi {suhu_pred:.1f}°C (THI: {thi_pred}). "
            "Nyalakan kipas sirkulasi untuk menjaga suhu kandang tetap adem agar nafsu makan ayam tidak turun."
        )
    else:
        heat_status = "NYAMAN (AMAN)"
        fan_action = "VENTILASI NORMAL / MINIMAL"
        recs.append(
            f"[KONDISI NYAMAN] Suhu prediksi {suhu_pred:.1f}°C (THI: {thi_pred}). "
            "Suhu ideal untuk ayam pejantan (tahan dingin, tidak perlu pemanas). "
            "Kipas cukup mode sirkulasi normal."
        )

    return {
        "thi_forecast": thi_pred,
        "heat_stress_status": heat_status,
        "fan_recommendation": fan_action,
        "messages": recs,
    }
