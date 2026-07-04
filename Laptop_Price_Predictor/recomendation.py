import pandas as pd


def load_data():
    df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")
    return df


def currency_conversion(df):
    df["Price_INR"] = df["Price_Euro"] * 110

    return df


def main():
    df = load_data()
    df = currency_conversion(df)


if __name__ == "__main__":
    main()
