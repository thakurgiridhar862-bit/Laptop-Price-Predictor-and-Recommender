import pandas as pd
from sklearn.model_selection import train_test_split


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


def main():
    df = load_data()
    data_overview(df)
    X, Y = feature_selection(df)
    X = encoding(X)
    X_train, X_test, Y_train, Y_test = tts(X, Y)


if __name__ == "__main__":
    main()
