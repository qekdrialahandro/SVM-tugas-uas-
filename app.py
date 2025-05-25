import streamlit as st
import joblib
import numpy as np

# Muat model dan scaler yang sudah disimpan
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Judul aplikasi
st.title('Prediksi Penyakit Diabetes')
st.write("Masukkan data pasien untuk mengetahui kemungkinan diabetes.")

# Input data pengguna
preg = st.number_input('Jumlah Kehamilan', min_value=0, step=1)
glucose = st.number_input('Kadar Glukosa', min_value=0)
bp = st.number_input('Tekanan Darah', min_value=0)
skin = st.number_input('Ketebalan Kulit', min_value=0)
insulin = st.number_input('Insulin', min_value=0)
bmi = st.number_input('BMI', min_value=0.0, format="%.2f")
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f")
age = st.number_input('Usia', min_value=0, step=1)

# Tombol Prediksi
if st.button('Prediksi'):
    # Siapkan data input
    data_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data_input = scaler.transform(data_input)  # Normalisasi data input
    prediction = model.predict(data_input)

    # Menampilkan hasil prediksi
    if prediction[0] == 1:
        st.error('Pasien berisiko terkena Diabetes! 🚨')
    else:
        st.success('Pasien tidak terkena Diabetes. 😊')
