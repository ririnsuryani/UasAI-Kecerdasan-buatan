# 1. Import Library
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# 2. Load Data (Contoh manual dari soal)
data = {
    'Lingkar_Batang': [0.3, 0.18, 0.46, 0.63, 0.23, 0.56, 0.39, 0.41, 0.62, 0.43, 0.15,
                       0.19, 0.17, 0.17, 0.22, 0.45, 0.39, 0.42, 0.38, 0.3, 0.18],
    'Tinggi': [7.21, 5.12, 8.83, 12.08, 5.81, 13.5, 10.9, 6.79, 10.66, 10.5, 2.67,
               20.34, 19.72, 19.8, 23.7, 32.51, 26.23, 32.51, 29.18, 26.1, 21.51],
    'Jenis_Pinus': ['Douglas Fir']*11 + ['White Pine']*10
}
df = pd.DataFrame(data)

# 3. Preprocessing Data
# Encoding Label (Text -> Numerik)
le = LabelEncoder()
df['Jenis_Pinus_Encoded'] = le.fit_transform(df['Jenis_Pinus'])  # Douglas Fir=0, White Pine=1

# Normalisasi Fitur (MinMax Scaling)
scaler = MinMaxScaler()
X = scaler.fit_transform(df[['Lingkar_Batang', 'Tinggi']])
y = df['Jenis_Pinus_Encoded']

# 4. Training Model KNN (K=11)
model = KNeighborsClassifier(n_neighbors=11)
model.fit(X, y)

# 5. Prediksi Data Uji
X_test = np.array([[0.4, 18.2]])  # Data uji: Lingkar Batang=0.2, Tinggi=15.2
X_test_scaled = scaler.transform(X_test)
prediction = model.predict(X_test_scaled)
predicted_label = le.inverse_transform(prediction)[0]

# 6. Tampilkan Hasil Prediksi
print("=== HASIL PREDIKSI ===")
print(f"Input Data: Lingkar Batang = {X_test[0][0]} m, Tinggi = {X_test[0][1]} m")
print(f"Jenis Pinus yang Diprediksi: {predicted_label}")
print("\nDetail:")
print(f"- Probabilitas Douglas Fir: {model.predict_proba(X_test_scaled)[0][0]:.2f}")
print(f"- Probabilitas White Pine: {model.predict_proba(X_test_scaled)[0][1]:.2f}")