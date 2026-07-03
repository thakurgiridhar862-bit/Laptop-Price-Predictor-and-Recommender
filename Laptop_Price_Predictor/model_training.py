import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
import os


def load_data():
    df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")
    return df


def data_overview(df):
    print(f"\nTotal row              : {df.shape[0]}")
    print(f"Total Columns          : {df.shape[1]}")
    print(f"Total Null Values      : {df.isnull().sum().sum()}")
    print(f"Total Duplicate Values : {df.duplicated().sum()}")
    print(" \n DATA TYPES")
    print("-" * 50)
    print(df.dtypes)


def feature_selection(df):

    X = df.drop(columns=["Price_Euro", "Laptop", "Model"])
    Y = df["Price_Euro"]

    print("\nFEATURE SELECTION")
    print("-" * 50)

    print("\nInput Features (X):")
    print(X.columns.tolist())

    print("\nTarget Variable (Y):")
    print(Y.name)

    return X, Y


def encoding(X):
    cat_cols = ["Status", "Brand", "CPU", "Storage_Type", "GPU", "Touch"]

    print("\nCategorical Columns:")
    print(cat_cols)

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    print("\nEncoding Completed Successfully")
    print(f"Shape After Encoding : {X.shape}")

    return X


def tts(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=56
    )
    print("\nTRAIN TEST SPLIT OF DATA")
    print("-" * 50)
    print(f"Shape of X_train : {X_train.shape}")
    print(f"Shape of X_test  : {X_test.shape}")
    print(f"Shape of Y_train : {Y_train.shape}")
    print(f"Shape of Y_test  : {Y_test.shape}")
    return X_train, X_test, Y_train, Y_test


def train_model(X_train, X_test, y_train):
    lr = LinearRegression()

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    print("\nMODEL TRAINING COMPLETED")
    print("-" * 50)
    print("First 10 Predictions:")
    print(y_pred[:10])

    return lr, y_pred


def model_evaluation(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nMODEL EVALUATION")
    print("-" * 50)
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")


def save_model(model):
    os.makedirs("Laptop_Price_Predictor/models", exist_ok=True)

    joblib.dump(model, "Laptop_Price_Predictor/models/laptop_price_model.pkl")

    print("\nModel saved successfully!")
    print("Path: Laptop_Price_Predictor/models/laptop_price_model.pkl")


def main():
    df = load_data()
    data_overview(df)
    X, Y = feature_selection(df)
    X = encoding(X)
    X["Screen"] = X["Screen"].fillna(X["Screen"].median())
    X_train, X_test, Y_train, Y_test = tts(X, Y)
    model, y_pred = train_model(X_train, X_test, Y_train)

    model_evaluation(Y_test, y_pred)
    save_model(model)


if __name__ == "__main__":
    main()
