import pandas as pd
import streamlit as st
import numpy as np
from sklearn.utils.validation import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from numpy import array

df = pd.read_excel("https://raw.githubusercontent.com/ZuniAmandaDewi/dataset/main/gizi.xlsx")

#ambil data
X = df.drop(columns="Jenis")  #data testing
y = df.Jenis #data class

#sidebar dengan radio
load_data, preprocessing, modelling, implementasi = st.tabs(["Load Data","Prepocessing", "Modeling", "Implementasi"])
#menu = st.sidebar.radio("Menu", ["Home","Load Data", "Preprocessing", "Modelling", "Implementasi"])

st.sidebar.title("Klasifikasi Status Gizi Balita")
st.sidebar.caption('https://raw.githubusercontent.com/ZuniAmandaDewi/dataset/main/gizi.xlsx')
st.sidebar.text("""
        Klasifikasi ini menggunakan data:
        1. Berat Badan  : Min = 3,4
                          Max = 28,8
        2. Tinggi Badan : Min = 50,5
                          Max = 115,0

        Hasil Klasifikasi:
        * Rendah
        * Normal
        """)

##halaman load data
with load_data :
    st.title("Data Asli")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        data = pd.read_excel(uploaded_file)
        st.dataframe(data)

##halaman Preprocessing
with preprocessing :
    st.title("Preprocessing")
    normalisasi = st.multiselect ("Pilih apa yang ingin Anda lakukan :", ["Normalisasi"])
    submit = st.button("Submit")
    if submit :
        if normalisasi :
            #ambil data
            X = data.drop(columns="Jenis")  #data testing
            y = data.Jenis #data class
                    
            #mengambil nama kolom
            judul = X.columns.copy() 

            #menghitung hasil normalisasi + menampilkan
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            hasil = pd.DataFrame(X,columns=judul)
            st.dataframe(hasil)

with modelling :
    st.title("Modelling")

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    metode = Perceptron(tol=1e-2, random_state=0)
    metode.fit(X_train, y_train)

    st.write ("Pilih metode yang ingin anda gunakan :")
    met = st.checkbox("Perceptron")
    if met :
        st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode.score(X_train, y_train)))
        st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode.score(X_test, y_test))))
    submit2 = st.button("Pilih")

    if submit2:      
        if met :
            st.write("Metode yang Anda gunakan Adalah Perceptron")
        else :
            st.write("Anda Belum Memilih Metode")

with implementasi :
        # section output
    def submit3():
        # input
        inputs = np.array([[bb, tb]])
        scaler = MinMaxScaler()
        scaler.fit(inputs)
        X = scaler.transform(inputs)
        st.subheader("Data yang Anda Inputkan :")
        st.write(inputs)

        # import label encoder
        le = joblib.load("le.save")

        # create output
        if met:
            metode = joblib.load("perceptron.joblib")
            y_pred = metode.predict(X)
            st.title("Perceptron")
            st.write(f"Data yang Anda masukkan tergolong dalam kelas : {le.inverse_transform(y_pred)[0]}")
        else :
            st.write("Metode yang Anda Pilih Belum Ada, Silahkan Kembali ke Tabs Modelling Untuk memilih Metode")

    st.title("Form Cek Status Gizi Balita")
    bb = st.number_input("Berat Badan", 3.4, 28.8, step=0.1)
    tb = st.number_input("Tinggi Badan", 50.5, 115.0, step=0.1)

    # create button submit
    submitted = st.button("Cek")
    if submitted:
        submit3()
