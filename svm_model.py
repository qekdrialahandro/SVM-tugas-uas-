import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv('diabetes.csv')  # Ganti dengan lokasi dataset Anda

# Pisahkan fitur dan label
X = df.drop(columns='Outcome')  # Fitur
y = df['Outcome']  # Label

# Pembagian data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat dan melatih model SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Simpan model
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
