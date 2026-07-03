import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Laptop_Price_Predictor/models/laptop_price_model.pkl")
model_columns = joblib.load("Laptop_Price_Predictor/models/model_columns.pkl")

df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("Laptop Price Predictor")
st.write("Select laptop specifications and estimate the laptop price.")

st.info("This app predicts laptop price using a Multiple Linear Regression model.")

st.subheader("Basic Details")

col1, col2 = st.columns(2)

with col1:
    status = st.selectbox("Status", sorted(df["Status"].dropna().unique()))
    brand = st.selectbox("Brand", sorted(df["Brand"].dropna().unique()))
    cpu = st.selectbox("CPU", sorted(df["CPU"].dropna().unique()))

with col2:
    storage_type = st.selectbox(
        "Storage Type", sorted(df["Storage_Type"].dropna().unique())
    )
    gpu = st.selectbox("GPU", sorted(df["GPU"].dropna().unique()))
    touch = st.selectbox("Touch Screen", sorted(df["Touch"].dropna().unique()))

st.subheader("Hardware Details")

col3, col4, col5 = st.columns(3)

with col3:
    ram = st.selectbox("RAM (GB)", sorted(df["RAM"].dropna().unique()))

with col4:
    storage = st.selectbox("Storage (GB)", sorted(df["Storage"].dropna().unique()))

with col5:
    screen = st.selectbox("Screen Size", sorted(df["Screen"].dropna().unique()))

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

euro_to_inr = 90

if st.button("Predict Price"):
    predicted_price_euro = model.predict(input_data)[0]
    predicted_price_inr = predicted_price_euro * euro_to_inr

    st.success("Prediction completed successfully!")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("Estimated Price (Euro)", f"€{predicted_price_euro:.2f}")

    with result_col2:
        st.metric("Estimated Price (INR)", f"₹{predicted_price_inr:,.2f}")

    st.write("Selected Configuration:")
    st.dataframe(input_data)
