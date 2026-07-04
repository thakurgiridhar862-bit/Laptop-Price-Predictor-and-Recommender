import pandas as pd


def load_data():
    df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")
    return df


def currency_conversion(df):
    df["Price_INR"] = df["Price_Euro"] * 110

    return df


def budget_filter(df, budget):
    fil_df = df[df["Price_INR"] <= budget]

    print("\nBudget Filter Applied")
    print("-" * 50)
    print(f"Budget : ₹{budget}")
    print(f"Matching Laptops : {fil_df.shape[0]}")

    return fil_df


def ram_filter(rec_df, ram):
    fil_df = rec_df[rec_df["RAM"] >= ram]

    print("\nRAM Filter Applied")
    print("-" * 50)
    print(f"Preffered RAM : {ram} GB")
    print(f"Matching Laptops : {fil_df.shape[0]}")

    return fil_df


def main():
    df = load_data()
    df = currency_conversion(df)
    rec_df = budget_filter(df, budget)
    rec_df = ram_filter(df, ram)


if __name__ == "__main__":
    main()
