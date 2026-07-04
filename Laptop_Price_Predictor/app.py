import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Laptop_Price_Predictor/models/laptop_price_model.pkl")
model_columns = joblib.load("Laptop_Price_Predictor/models/model_columns.pkl")

df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

st.sidebar.title("About Project")
st.sidebar.write(
    "This project predicts laptop prices using a trained Multiple Linear Regression model."
)
st.subheader("Selected Configuration")
st.sidebar.write("Model Score: R2 = 0.8205")
st.sidebar.write("Currency: Euro and INR")
st.sidebar.write("Dataset: 2160 laptop records")
st.sidebar.write("Model: Multiple Linear Regression")

st.sidebar.subheader("Project Steps")
st.sidebar.write("Data Cleaning")
st.sidebar.write("EDA")
st.sidebar.write("Model Training")
st.sidebar.write("Scratch Model")
st.sidebar.write("Streamlit App")

st.title("Laptop Price Predictor")
st.write(
    "A machine learning application to estimate laptop prices based on specifications."
)

st.divider()

st.subheader("Select Laptop Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    status = st.selectbox("Laptop Status", sorted(df["Status"].dropna().unique()))
    brand = st.selectbox("Brand", sorted(df["Brand"].dropna().unique()))
    cpu = st.selectbox("Processor", sorted(df["CPU"].dropna().unique()))

with col2:
    ram = st.selectbox("RAM (GB)", sorted(df["RAM"].dropna().unique()))
    storage = st.selectbox("Storage (GB)", sorted(df["Storage"].dropna().unique()))
    storage_type = st.selectbox(
        "Storage Type", sorted(df["Storage_Type"].dropna().unique())
    )

with col3:
    gpu = st.selectbox("Graphics Card", sorted(df["GPU"].dropna().unique()))
    screen = st.selectbox("Screen Size", sorted(df["Screen"].dropna().unique()))
    touch = st.selectbox("Touch Screen", sorted(df["Touch"].dropna().unique()))

st.divider()

input_data = pd.DataFrame(
    {
        "Status": [status],
        "Brand": [brand],
        "CPU": [cpu],
        "RAM": [ram],
        "Storage": [storage],
        "Storage_Type": [storage_type],
        "GPU": [gpu],
        "Screen": [screen],
        "Touch": [touch],
    }
)

st.subheader("Selected Configuration")
st.dataframe(input_data, use_container_width=True)

input_encoded = pd.get_dummies(input_data, drop_first=True)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

euro_to_inr = 110

predict_button = st.button("Predict Laptop Price")

if predict_button:
    predicted_price_euro = model.predict(input_encoded)[0]
    predicted_price_inr = predicted_price_euro * euro_to_inr

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("Estimated Price in Euro", f"€{predicted_price_euro:.2f}")

    with result_col2:
        st.metric("Estimated Price in INR", f"₹{predicted_price_inr:,.2f}")

    st.info(
        "The predicted price is an estimate based on the available dataset and selected specifications."
    )
