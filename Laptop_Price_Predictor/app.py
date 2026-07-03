import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Laptop_Price_Predictor/models/laptop_price_model.pkl")
model_columns = joblib.load("Laptop_Price_Predictor/models/model_columns.pkl")

df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("Laptop Price Predictor")
st.write("Enter laptop details and predict the estimated price.")

st.subheader("Laptop Details")

status = st.selectbox("Status", sorted(df["Status"].dropna().unique()))
brand = st.selectbox("Brand", sorted(df["Brand"].dropna().unique()))
cpu = st.selectbox("CPU", sorted(df["CPU"].dropna().unique()))
storage_type = st.selectbox(
    "Storage Type", sorted(df["Storage_Type"].dropna().unique())
)
gpu = st.selectbox("GPU", sorted(df["GPU"].dropna().unique()))
touch = st.selectbox("Touch Screen", sorted(df["Touch"].dropna().unique()))

ram = st.number_input("RAM (GB)", min_value=1, max_value=128, value=8)
storage = st.number_input("Storage (GB)", min_value=32, max_value=4000, value=512)
screen = st.number_input("Screen Size", min_value=10.0, max_value=20.0, value=15.6)

input_data = pd.DataFrame(
    {
        "Status": [status],
        "Brand": [brand],
        "CPU": [cpu],
        "Storage_Type": [storage_type],
        "GPU": [gpu],
        "Touch": [touch],
        "RAM": [ram],
        "Storage": [storage],
        "Screen": [screen],
    }
)

input_data = pd.get_dummies(input_data, drop_first=True)

input_data = input_data.reindex(columns=model_columns, fill_value=0)

if st.button("Predict Price"):
    prediction = model.predict(input_data)

    st.success(f"Estimated Laptop Price: €{prediction[0]:.2f}")
