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


def brand_analysis(df):

    print("\nBRAND ANALYSIS")
    print("-" * 50)

    print(f"Total Brands : {df['Brand'].nunique()}")
    print("Brand counts")
    print("-" * 50)
    br_count = df["Brand"].value_counts()
    print(br_count)
    print(f"Most Common Brand : {br_count.index[0]} ({br_count.iloc[0]} laptops)")

    # graph 1
    plt.figure(figsize=(16, 5))
    plt.title("Top 10 Laptop Brands")
    plt.xlabel("Brands")
    plt.ylabel("Number of laptops")
    plt.grid(axis="y", alpha=0.3)
    X = br_count.head(10).index
    Y = br_count.head(10).values
    ax1 = sns.barplot(
        x=X,
        y=Y,
        palette="crest",
    )
    for i in ax1.containers:
        ax1.bar_label(i, padding=5)
    plt.savefig(
        "Laptop_Price_Predictor/graphs/top10_brands.png", dpi=300, bbox_inches="tight"
    )

    avg_brand_price = (
        df.groupby("Brand")["Price_Euro"].mean().round(2).sort_values(ascending=False)
    )
    print("Average Price by brand")
    print("-" * 50)
    print(avg_brand_price)
    # graph 2
    plt.figure(figsize=(16, 5))
    plt.title("Average Laptop Price by Brand")
    plt.xlabel("Brand")
    plt.ylabel("Average Price (Euro)")
    plt.grid(axis="y", alpha=0.3)
    X = avg_brand_price.head(10).index
    Y = avg_brand_price.head(10).values
    ax2 = sns.barplot(
        x=X,
        y=Y,
        palette="rocket",
    )
    for i in ax2.containers:
        ax2.bar_label(i, padding=5)
    plt.savefig(
        "Laptop_Price_Predictor/graphs/average_price_by_brand.png",
        dpi=300,
        bbox_inches="tight",
    )


def Ram_analysis(df):

    print("\nRAM ANALYSIS")
    print("-" * 50)

    print(f"Total RAM Variants : {df['RAM'].nunique()}")
    print("RAM counts")
    print("-" * 50)
    ram_count = df["RAM"].value_counts()
    print(ram_count)
    avg_ram_price = (
        df.groupby("RAM")["Price_Euro"].mean().round(2).sort_values(ascending=False)
    )
    print("Average Price by RAM")
    print("-" * 50)
    print(avg_ram_price)
    plt.figure(figsize=(16, 5))
    plt.title("Average Laptop Price by RAM")
    plt.xlabel("RAM (GB)")
    plt.ylabel("Average Price (Euro)")
    plt.grid(axis="y", alpha=0.3)
    X = avg_ram_price.index
    Y = avg_ram_price.values
    ax = sns.barplot(
        x=X,
        y=Y,
        palette="magma",
    )
    for i in ax.containers:
        ax.bar_label(i, padding=5)
    plt.savefig(
        "Laptop_Price_Predictor/graphs/average_price_by_ram.png",
        dpi=300,
        bbox_inches="tight",
    )


def storage_analysis(df):

    print("\nSTORAGE TYPE ANALYSIS")
    print("-" * 50)

    print(f"Total Storage Type Variants : {df['Storage_Type'].nunique()}")
    print("Storage Type counts")
    print("-" * 50)
    storage_count = df["Storage_Type"].value_counts()
    print(storage_count)
    avg_storage_price = (
        df.groupby("Storage_Type")["Price_Euro"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )
    print("Average Price by Storage Type")
    print("-" * 50)
    print(avg_storage_price)
    plt.figure(figsize=(16, 5))
    plt.title("Average Laptop Price by Storage Type")
    plt.xlabel("Storage Type")
    plt.ylabel("Average Price (Euro)")
    plt.grid(axis="y", alpha=0.3)
    X = avg_storage_price.index
    Y = avg_storage_price.values
    ax = sns.barplot(
        x=X,
        y=Y,
        palette="viridis",
    )
    for i in ax.containers:
        ax.bar_label(i, padding=5)
    plt.savefig(
        "Laptop_Price_Predictor/graphs/average_price_by_storage_type.png",
        dpi=300,
        bbox_inches="tight",
    )


def cpu_analysis(df):

    print("\nCPU TYPE ANALYSIS")
    print("-" * 50)

    print(f"Total CPU Type Variants : {df['CPU'].nunique()}")

    print("\nCPU Type Counts")
    print("-" * 50)
    cpu_count = df["CPU"].value_counts()
    print(cpu_count)

    # Graph
    plt.figure(figsize=(16, 5))
    plt.title("Top 10 Most Common CPUs")
    plt.xlabel("CPU Type")
    plt.ylabel("Number of Laptops")
    plt.grid(axis="y", alpha=0.3)

    X = cpu_count.head(10).index
    Y = cpu_count.head(10).values

    ax = sns.barplot(
        x=X,
        y=Y,
        palette="coolwarm",
    )

    for container in ax.containers:
        ax.bar_label(container, padding=5)

    plt.savefig(
        "Laptop_Price_Predictor/graphs/top10_common_cpus.png",
        dpi=300,
        bbox_inches="tight",
    )


def gpu_analysis(df):

    print("\nGPU TYPE ANALYSIS")
    print("-" * 50)

    print(f"Total GPU Type Variants : {df['GPU'].nunique()}")

    print("\nGPU Type Counts")
    print("-" * 50)
    gpu_count = df["GPU"].value_counts()
    print(gpu_count)

    # Graph
    plt.figure(figsize=(16, 5))
    plt.title("Top 10 Most Common GPUs")
    plt.xlabel("GPU Type")
    plt.ylabel("Number of Laptops")
    plt.grid(axis="y", alpha=0.3)

    gpu_count = gpu_count.drop("Unknown")

    X = gpu_count.head(10).index
    Y = gpu_count.head(10).values

    ax = sns.barplot(
        x=X,
        y=Y,
        palette="Set2",
    )

    for container in ax.containers:
        ax.bar_label(container, padding=5)

    plt.savefig(
        "Laptop_Price_Predictor/graphs/top10_common_gpus.png",
        dpi=300,
        bbox_inches="tight",
    )


def touchscreen_analysis(df):

    print("\nTOUCHSCREEN ANALYSIS")
    print("-" * 50)

    print("Touchscreen Counts")
    print("-" * 50)

    touch_count = df["Touch"].value_counts()
    print(touch_count)

    plt.figure(figsize=(8, 5))
    plt.title("Touchscreen vs Non-Touchscreen Laptops")
    plt.xlabel("Touchscreen")
    plt.ylabel("Number of Laptops")
    plt.grid(axis="y", alpha=0.3)

    X = touch_count.index
    Y = touch_count.values

    ax = sns.barplot(
        x=X,
        y=Y,
        palette="Pastel1",
    )

    for container in ax.containers:
        ax.bar_label(container, padding=5)

    plt.savefig(
        "Laptop_Price_Predictor/graphs/touchscreen_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )


def save_cleaned_data(df):
    df.to_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv", index=False)
    print("\nCleaned dataset saved successfully!")


def main():
    df = load_data()
    data_overview(df)
    df = clean_data(df)
    price_dist(df)
    brand_analysis(df)
    Ram_analysis(df)
    storage_analysis(df)
    cpu_analysis(df)
    gpu_analysis(df)
    touchscreen_analysis(df)
    save_cleaned_data(df)


if __name__ == "__main__":
    main()
