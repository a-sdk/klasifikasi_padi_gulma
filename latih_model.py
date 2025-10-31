import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("Memuat data latih dari CSV...")
df = pd.read_csv('Hasil/data_training.csv')
X = df.drop('label', axis=1).values # Ambil semua kolom KECUALI 'label'
Xcol = df.drop('label', axis=1)
y = df['label']              # Ambil hanya kolom 'label'

print(f"Data dimuat. Jumlah sampel: {X.shape[0]}, Jumlah fitur: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Data dibagi:")
print(f"  - Jumlah sampel Latih (Train): {X_train.shape[0]}")
print(f"  - Jumlah sampel Tes (Test):   {X_test.shape[0]}")

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
print("Model selesai dilatih!")

print("Mengevaluasi model menggunakan Test Set...")
y_pred = model.predict(X_test)

print("\n--- HASIL EVALUASI MODEL ---")

# 1. Akurasi Keseluruhan
akurasi = accuracy_score(y_test, y_pred) * 100
print(f"Akurasi Keseluruhan: {akurasi:.2f}%")

# 2. Laporan Klasifikasi (Sangat Penting)
print("\nLaporan Klasifikasi:")
# Ganti 'Padi (1)' dan 'Gulma (2)' sesuai label Anda
print(classification_report(y_test, y_pred, target_names=['Padi (1)', 'Gulma (2)']))

# 3. Confusion Matrix (Sangat Berguna)
print("\nConfusion Matrix:")
print("         Prediksi Padi | Prediksi Gulma")
cm = confusion_matrix(y_test, y_pred)
print(f"Aktual Padi   {cm[0]}")
print(f"Aktual Gulma  {cm[1]}")

# Tentukan label untuk sumbu
labels_kelas = ['Padi (1)', 'Gulma (2)']

# Buat gambar (figure) matplotlib
plt.figure(figsize=(8, 6))

# Buat heatmap menggunakan Seaborn
sns.heatmap(
    cm, 
    annot=True,     # Menampilkan angka di dalam kotak
    fmt='d',        # Format angka sebagai integer (bilangan bulat)
    cmap='Blues',   # Skema warna (biru adalah umum untuk ini)
    xticklabels=labels_kelas, 
    yticklabels=labels_kelas
)

# Tambahkan judul dan label sumbu
plt.title('Confusion Matrix Heatmap', fontsize=16)
plt.ylabel('Label Aktual (Sebenarnya)', fontsize=12)
plt.xlabel('Label Prediksi (Tebakan Model)', fontsize=12)

# Tampilkan plot
plt.show()

print("\nFitur Terpenting Menurut Model:")
    
# 1. Dapatkan skor pentingnya dari model yang sudah dilatih
importances = model.feature_importances_

# 2. Dapatkan nama-nama fitur dari data 'X' Anda
feature_names = Xcol.columns

# 3. Gabungkan keduanya ke dalam DataFrame Pandas agar mudah dibaca
feature_importance_df = pd.DataFrame({
    'Fitur': feature_names, 
    'Pentingnya': importances
}).sort_values(by='Pentingnya', ascending=False) # Urutkan dari paling penting

# 4. Tampilkan 10 fitur teratas
print(feature_importance_df.head(10)) 

# (Opsional) Tampilkan sebagai grafik batang
plt.figure(figsize=(10, 8))
sns.barplot(x='Pentingnya', y='Fitur', data=feature_importance_df.head(10))
plt.title('10 Fitur Terpenting')
plt.xlabel('Skor Pentingnya')
plt.ylabel('Fitur')
plt.show()

# Menyimpan model
model_filename = 'model_random_forest.joblib'
print(f"Menyimpan model yang sudah dilatih ke {model_filename}...")
joblib.dump(model, model_filename)
print("Model berhasil disimpan.")