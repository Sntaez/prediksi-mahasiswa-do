import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data_mahasiswa_1000.csv")

data = load_data()

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Informasi Data", "Visualisasi", "Statistik Deskriptif", "Inference"])

# Halaman: Informasi Data
if page == "Informasi Data":
    st.title("Dashboard Analisis Data Mahasiswa")
    st.subheader("Dataset Mahasiswa")
    st.write(data.head())
    st.write("Jumlah baris dan kolom:", data.shape)
    st.write("Kolom:", data.columns.tolist())

# Halaman: Visualisasi
elif page == "Visualisasi":
    st.title("Visualisasi Data")

    st.subheader("Jumlah Mahasiswa per Semester Aktif")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="semester_aktif", data=data, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Relasi Kehadiran vs Retake Count")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=data, x="kehadiran_total", y="retake_count", hue="dropout", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Aktivitas E-learning vs Dropout")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=data, x="dropout", y="elearning_login_count", ax=ax3)
    st.pyplot(fig3)

# Halaman: Statistik Deskriptif
elif page == "Statistik Deskriptif":
    st.title("Statistik Deskriptif")
    st.write(data.describe(include='all'))

    st.markdown("### Korelasi Numerik")
    num_data = data.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots()
    sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Halaman: Inference
elif page == "Inference":
    st.title("Prediksi Dropout Mahasiswa")

    st.markdown("### Masukkan Data Mahasiswa")
    input_data = {}
    numeric_cols = [
        'semester_aktif', 'kehadiran_total', 'retake_count',
        'elearning_login_count', 'elearning_partisipasi',
        'elearning_tugas_dikumpul', 'beban_kerja', 'pendapatan_ortu',
        'ipk_sem1', 'ipk_sem2', 'ipk_sem3', 'ipk_sem4',
        'ipk_sem5', 'ipk_sem6', 'ipk_sem7', 'ipk_sem8'
    ]
    categorical_cols = ['status_kerja', 'status_beasiswa']

    for col in numeric_cols:
        input_data[col] = st.number_input(col, value=0.0)

    input_data['status_kerja'] = st.selectbox('Status Kerja', ['Bekerja', 'Tidak Bekerja'])
    input_data['status_beasiswa'] = st.selectbox('Status Beasiswa', ['Ya', 'Tidak'])

    # Load model
    if st.button("Prediksi Dropout"):
        model = joblib.load("model.pkl")  # Pastikan model Anda telah disimpan sebagai 'model.pkl'
        
        # Preprocess input
        df_input = pd.DataFrame([input_data])
        df_input['status_kerja'] = df_input['status_kerja'].map({'Bekerja': 1, 'Tidak Bekerja': 0})
        df_input['status_beasiswa'] = df_input['status_beasiswa'].map({'Ya': 1, 'Tidak': 0})
        
        pred = model.predict(df_input)[0]
        result = "BERPOTENSI DROPOUT" if pred == 1 else "TIDAK BERDROPOUT"
        st.success(f"Hasil Prediksi: {result}")
