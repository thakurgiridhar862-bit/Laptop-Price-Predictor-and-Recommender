import pandas as pd


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
    X = df.drop(columns=["Price_Euro"])
    Y = df["Price_Euro"]

    print("\nFEATURE SELECTION")
    print("-" * 50)

    print("\nInput Features (X):")
    print(X.columns.tolist())

    print("\nTarget Variable (y):")
    print(Y.name)

    return X, Y


def main():
    df = load_data()
    data_overview(df)
    X, Y = feature_selection(df)


if __name__ == "__main__":
    main()
