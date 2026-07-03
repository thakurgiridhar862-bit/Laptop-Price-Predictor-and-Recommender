import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


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


def main():
    df = load_data()
    data_overview(df)
    X, Y = feature_selection(df)
    X = encoding(X)


if __name__ == "__main__":
    main()
