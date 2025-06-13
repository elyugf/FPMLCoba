import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan encoder
try:
    model = joblib.load('ann_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    unique_categories = joblib.load('unique_categories.pkl')

    gender_options = unique_categories['gender']
    occupation_options = unique_categories['occupation']
    bmi_options = unique_categories['bmi']
except Exception as e:
    st.error(f"Gagal memuat model atau encoder: {e}")
    st.stop()

st.title("Prediksi Gangguan Tidur (ANN - scikit-learn)")
st.write("Masukkan informasi Anda:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", gender_options)
    age = st.slider("Usia", 18, 90, 30)
    occupation = st.selectbox("Pekerjaan", occupation_options)
    sleep_duration = st.slider("Durasi Tidur (jam)", 4.0, 10.0, 7.0, 0.1)
    quality_of_sleep = st.slider("Kualitas Tidur (1-10)", 1, 10, 7)
    physical_activity_level = st.slider("Aktivitas Fisik (1-100)", 1, 100, 50)

with col2:
    stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
    bmi_category = st.selectbox("Kategori BMI", bmi_options)
    systolic_bp = st.number_input("Tekanan Darah Sistolik", 80, 200, 120)
    diastolic_bp = st.number_input("Tekanan Darah Diastolik", 40, 150, 80)
    heart_rate = st.number_input("Detak Jantung", 40, 120, 70)
    daily_steps = st.number_input("Langkah Harian", 1000, 15000, 5000)

if st.button("Prediksi Gangguan Tidur"):
    input_dict = {
        'Gender': [gender],
        'Age': [age],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity_level],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_category],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps]
    }

    input_df = pd.DataFrame(input_dict)

    # One-hot encode sesuai dengan training
    input_encoded = pd.get_dummies(input_df)

    # Pastikan semua kolom yang digunakan model ada
    missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0

    # Urutkan kolom sesuai model
    input_encoded = input_encoded[model.feature_names_in_]

    try:
        pred_encoded = model.predict(input_encoded)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"Prediksi gangguan tidur Anda: **{pred_label}**")

        if pred_label == "Sleep Apnea":
            st.warning("Konsultasikan dengan dokter untuk evaluasi lanjutan.")
        elif pred_label == "Insomnia":
            st.info("Coba teknik relaksasi, kebiasaan tidur sehat, atau konsultasi medis.")
        else:
            st.balloons()
            st.success("Tidur Anda tampaknya sehat! 🎉")

    except Exception as e:
        st.error(f"Kesalahan saat memproses input: {e}")

import seaborn as sns
import matplotlib.pyplot as plt

# Load hasil evaluasi model
try:
    conf_matrix = joblib.load('conf_matrix.pkl')
    class_report = joblib.load('classification_report.pkl')
except Exception as e:
    st.warning("Hasil evaluasi belum tersedia. Jalankan training terlebih dahulu.")
    conf_matrix = None
    class_report = None

# Setelah tampilkan prediksi:
if conf_matrix is not None and class_report is not None:
    st.markdown("## 🔍 Evaluasi Model")

    # Plot Confusion Matrix
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap='Blues', ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

    # Plot bar chart: precision per class
    st.markdown("### Precision per Kelas")
    class_names = list(class_report.keys())[:-3]  # skip avg/total
    precisions = [class_report[cls]['precision'] for cls in class_names]

    fig2, ax2 = plt.subplots()
    sns.barplot(x=precisions, y=class_names, ax=ax2, palette='Set2')
    ax2.set_xlabel("Precision")
    ax2.set_title("Precision per Kelas")
    st.pyplot(fig2)

    # Show classification report
    st.markdown("### Classification Report")
    report_df = pd.DataFrame(class_report).T.iloc[:-3]
    st.dataframe(report_df.style.format("{:.2f}"))
