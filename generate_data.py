import os
import numpy as np
import pandas as pd


def generate_aws_dataset(days: int = 45, output_path: str = "data/raw_history.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.default_rng(42)

    # Buat timestamp per 1 menit selama 45 hari terakhir (WIB)
    end_time = pd.Timestamp.now(tz="Asia/Jakarta").floor("h")
    start_time = end_time - pd.Timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq="1min")
    n = len(timestamps)

    hour_float = (timestamps.hour + timestamps.minute / 60.0).to_numpy()

    # Simulasi siklus diurnal suhu tropis Semarang (puncak siang ~33°C, malam ~24.5°C)
    suhu = 28.8 + 4.4 * np.sin((hour_float - 8.0) * np.pi / 12.0) + rng.normal(0, 0.35, n)

    # Kelembapan berbanding terbalik dengan suhu
    kelembapan = 74.0 - 15.0 * np.sin((hour_float - 8.0) * np.pi / 12.0) + rng.normal(0, 1.8, n)
    kelembapan = np.clip(kelembapan, 45.0, 98.0)

    # Tekanan atmosfer semidiurnal
    tekanan = 1010.5 + 1.5 * np.cos((hour_float - 10.0) * np.pi / 6.0) + rng.normal(0, 0.25, n)

    # Intensitas cahaya matahari (0 pada malam hari, puncak siang hari)
    solar_curve = np.maximum(0.0, np.sin((hour_float - 6.0) * np.pi / 12.0))
    cahaya = (solar_curve ** 1.3) * 78000.0 + rng.normal(0, 300, n)
    cahaya = np.where((hour_float < 5.8) | (hour_float > 18.2), 0.0, np.clip(cahaya, 0, 150000))

    # Kecepatan angin (km/jam)
    kecepatan_angin = np.abs(6.5 + 3.5 * np.sin((hour_float - 10.0) * np.pi / 12.0) + rng.normal(0, 1.5, n))

    # Simulasi kejadian hujan sporadis
    hujan = np.zeros(n)
    rain_events = rng.choice(n // 60, size=int(days * 1.4), replace=False)
    for hr_idx in rain_events:
        start_i = hr_idx * 60
        duration = rng.integers(20, 90)
        end_i = min(n, start_i + duration)
        intensity = rng.choice([0.02, 0.06, 0.14], p=[0.5, 0.35, 0.15])
        hujan[start_i:end_i] = rng.exponential(intensity, size=(end_i - start_i))
        suhu[start_i:end_i] -= rng.uniform(1.5, 3.5)
        kelembapan[start_i:end_i] = np.clip(kelembapan[start_i:end_i] + rng.uniform(8, 16), 0, 99.5)
        cahaya[start_i:end_i] *= 0.25

    epoch_ms = (timestamps - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    df = pd.DataFrame({
        "timestamp": epoch_ms,
        "suhu": np.round(suhu, 2),
        "kelembapan": np.round(kelembapan, 2),
        "tekanan": np.round(tekanan, 2),
        "cahaya": np.round(cahaya, 1),
        "hujan": np.round(hujan, 3),
        "kecepatan_angin": np.round(kecepatan_angin, 2),
    })

    # Sisipkan sedikit outlier & missing value untuk menguji pipeline pembersihan
    outlier_idx = rng.choice(n, size=15, replace=False)
    df.loc[outlier_idx[:5], "suhu"] = 999.0
    df.loc[outlier_idx[5:10], "kelembapan"] = -15.0
    df.loc[outlier_idx[10:], "tekanan"] = np.nan

    # Simulasikan 1x periode perangkat offline selama 3 jam
    df = df.drop(df.index[3000:3180]).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"[OK] Berhasil generate {len(df)} baris data sensor ke {output_path}")


if __name__ == "__main__":
    generate_aws_dataset()
