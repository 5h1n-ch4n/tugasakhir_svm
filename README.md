# AWS SVM Weather & Heat-Stress Forecasting (Kandang Ayam Pejantan)

Pipeline *Machine Learning* berbasis **Support Vector Machine (SVC & SVR)** yang berjalan otomatis di **GitHub Actions** untuk pembersihan data sensor, normalisasi (`StandardScaler`), prediksi cuaca 1 jam ke depan ($t+1$), dan perhitungan **THI (Temperature-Humidity Index)** khusus kandang **ayam pejantan**.

## Aturan Domain Kandang Ayam Pejantan (`config.py`)
1. **Tahan Suhu Dingin**: Ayam pejantan memiliki toleransi tinggi terhadap suhu dingin, sehingga sistem tidak memicu alarm pemanas (*brooding*) saat suhu turun.
2. **Prediksi Hujan Informatif**: Klasifikasi hujan (`Hujan Ringan`, `Hujan Lebat`, `Berang-berang`) ditampilkan murni sebagai informasi kondisi cuaca lingkungan sekitar kandang.
3. **Fokus Stres Panas (THI & Kontrol Kipas)**: Risiko utama ayam pejantan adalah *heat stress* (kepanasan) yang menurunkan nafsu makan. Ketika prediksi `suhu >= 29.5°C` atau `THI >= 78.0`, sistem merekomendasikan penyalaan kipas *exhaust/blower* agar kandang tetap adem.
