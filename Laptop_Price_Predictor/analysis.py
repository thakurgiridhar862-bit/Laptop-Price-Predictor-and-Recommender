import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    df = pd.read_csv("Laptop_Price_Predictor/data/laptops.csv")

    return df


def data_overview(df):
    print("-" * 60)
    print("DATASET OVERVIEW")
    print("-" * 50)

    print("\nFIRST FIVE ROWS")
    print(df.head(5))

    print(f"\nTotal row              : {df.shape[0]}")
    print(f"Total Columns          : {df.shape[1]}")
    print(f"Total Null Values      : {df.isnull().sum().sum()}")
    print(f"Total Duplicate Values : {df.duplicated().sum()}")

    print("\nMISSING VALUES ")
    print("-" * 50)
    print(df.isnull().sum())

    print(" \n DATA TYPES")
    print("-" * 50)
    print(df.dtypes)

    print("\nNUMERICAL SUMMARY")
    print("-" * 50)
    print(df.describe())


def clean_data(df):

    print("\n DATA CLEANING")
    print("-" * 50)

    df = df.rename(columns={"Storage type": "Storage_Type"})
    df = df.rename(columns={"Final Price": "Price_Euro"})
    df["Storage_Type"] = df["Storage_Type"].fillna(df["Storage_Type"].mode()[0])
    df["GPU"] = df["GPU"].fillna("Unknown")
    print("Data Cleaning Completed Successfully")
    print("-" * 50)
    print(f"After Cleaning Total Duplicate Values : {df.duplicated().sum()}")

    print("\nAFTER CLEANING MISSING VALUES ")
    print("-" * 50)
    print(df.isnull().sum())

    return df


def price_dist(df):
    print("\nPRICE DISTRIBUTION")
    print("-" * 50)

    print(f"Maximum Price (in Euros) : €{df['Price_Euro'].max()}")
    print(f"Minimum Price (in Euros) : €{df['Price_Euro'].min()}")
    print(f"Average Price (in Euros) : €{df['Price_Euro'].mean().round()}")
    print(f"Median Price (in Euros)  : €{df['Price_Euro'].median().round(2)}")

    mean_price = df["Price_Euro"].mean().round(2)
    median_price = df["Price_Euro"].median().round(2)
    plt.figure(figsize=(10, 6))
    plt.grid(axis="y", alpha=0.3)
    plt.title("Distribution of Laptop Prices")
    plt.xlabel("Price (Euro)")
    plt.ylabel("Number of Laptops")
    sns.histplot(df, x="Price_Euro", bins=25)
    plt.axvline(
        x=mean_price, color="red", linestyle="--", linewidth=2, label="Mean Price"
    )

    plt.axvline(
        x=median_price, color="green", linestyle="--", linewidth=2, label="Median Price"
    )

    plt.legend()
    plt.savefig(
        "Laptop_Price_Predictor/graphs/price_dist.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def main():
    df = load_data()
    data_overview(df)
    df = clean_data(df)
    price_dist(df)


if __name__ == "__main__":
    main()
