import pandas as pd


def load_data():
    df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")
    return df


def currency_conversion(df):
    df["Price_INR"] = df["Price_Euro"] * 110

    return df


def user_input(df):
    budget = int(input("Enter Your Budget (INR) : "))
    ram = int(input("Enter Minimum RAM (GB) : "))
    storage = int(input("Enter Minimum Storage (GB) : "))

    print("\nAvailable Brands")
    print("-" * 50)
    print(df["Brand"].unique())

    brand = input("\nEnter Preferred Brand : ")

    touch = input("\nTouch Screen (Yes/No) : ")

    return budget, ram, storage, brand, touch


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


def storage_filter(rec_df, storage):
    fil_df = rec_df[rec_df["Storage"] >= storage]

    print("\nStorage Filter Applied")
    print("-" * 50)
    print(f"Preffered Storage : {storage} GB")
    print(f"Matching Laptops  : {fil_df.shape[0]}")

    return fil_df


def brand_filter(rec_df, brand):
    fil_df = rec_df[rec_df["Brand"] == brand]

    print("\nBrand Filter Applied")
    print("-" * 50)
    print(f"Preffered Brand : {brand} GB")
    print(f"Matching Laptops  : {fil_df.shape[0]}")

    return fil_df


def main():
    df = load_data()
    df = currency_conversion(df)
    budget, ram, storage, brand, touch = user_input(df)
    rec_df = budget_filter(df, budget)
    rec_df = ram_filter(rec_df, ram)
    rec_df = storage_filter(rec_df, storage)
    rec_df = brand_filter(rec_df, brand)


if __name__ == "__main__":
    main()
