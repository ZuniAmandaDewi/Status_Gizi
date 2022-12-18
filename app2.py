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
load_data, preprocessing, modelling, implementasi = st.tabs(["Load Data","Prepocessing", "Akurasi", "Implementasi"])
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
    uploaded_files = st.file_uploader("Upload file Excel", accept_multiple_files=True)
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
    st.title("Akurasi Perceptron")

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train1,X_test1,y_train1,y_test1 = train_test_split(X,y,test_size=0.3,random_state=4)
    X_train2,X_test2,y_train2,y_test2 = train_test_split(X,y,test_size=0.2,random_state=4)
    X_train3,X_test3,y_train3,y_test3 = train_test_split(X,y,test_size=0.1,random_state=4)
    metode = Perceptron(tol=1e-2, random_state=0)
    metode.fit(X_train1, y_train1)
    metode.fit(X_train2, y_train2)
    metode.fit(X_train3, y_train3)

    st.write ("Pilih pembagian data yang ingin anda gunakan :")
    met1 = st.checkbox("70:30")
    if met1 :
        st.write("Hasil Akurasi Data Training 70% sebesar : ", (100 * metode.score(X_train1, y_train1)))
        st.write("Hasil Akurasi Data Testing 30% sebesar : ", (100 * (metode.score(X_test1, y_test1))))
    met2 = st.checkbox("80:20")
    if met2 :
        st.write("Hasil Akurasi Data Training 80% sebesar : ", (100 * metode.score(X_train2, y_train2)))
        st.write("Hasil Akurasi Data Testing 20% sebesar : ", (100 * metode.score(X_test2, y_test2)))
    met3 = st.checkbox("90:10")
    if met3 :
        st.write("Hasil Akurasi Data Training 90% sebesar : ", (100 * metode.score(X_train3, y_train3)))
        st.write("Hasil Akurasi Data Testing 10% sebesar : ", (100 * metode.score(X_test3, y_test3)))
    submit2 = st.button("Pilih")

    if submit2:      
        if met1 :
            st.write("Pembagian Data yang Anda gunakan Adalah training 70% dan testing 30%")

        elif met2 :
            st.write("Pembagian Data yang Anda gunakan Adalah training 80% dan testing 20%")

        elif met3 :
            st.write("Pembagian Data yang Anda gunakan Adalah training 90% dan testing 10%")

        else :
            st.write("Anda Belum Memilih Pembagian Data")


with implementasi :
        # section output
    def submit3():
        joblib.load("scaler.save")
        X = scaler.transform([[bb, tb]])
        # input
        inputs = np.array(X)
        st.subheader("Data yang Anda Inputkan :")
        st.write(inputs)

        # import label encoder
        le = joblib.load("le.save")

        # create output
        if met1:
            metode = joblib.load("perceptron.joblib")
            y_pred = metode.predict(inputs)
            st.title("Perceptron 70/30")
            st.write(f"Data yang Anda masukkan tergolong dalam kelas : {le.inverse_transform(y_pred)[0]}")
            
        elif met2:
            metode = joblib.load("perceptron.joblib")
            y_pred = metode.predict(inputs)
            st.title("Perceptron 80/20")
            st.write(f"Data yang Anda masukkan tergolong dalam kelas : {le.inverse_transform(y_pred)[0]}")

        elif met3:
            metode = joblib.load("perceptron.joblib")
            y_pred = metode.predict(inputs)
            st.title("Perceptron 80/20")
            st.write(f"Data yang Anda masukkan tergolong dalam kelas : {le.inverse_transform(y_pred)[0]}")

        else :
            st.write("Pembagian Data yang Anda Pilih Belum Ada, Silahkan Kembali ke Tabs Akurasi Untuk memilih Pembagian Data")

    st.title("Form Cek Status Gizi Balita")
    bb = st.number_input("Berat Badan", 3.4, 28.8, step=0.1)
    tb = st.number_input("Tinggi Badan", 50.5, 115.0, step=0.1)

    # create button submit
    submitted = st.button("Cek")
    if submitted:
        submit3()
