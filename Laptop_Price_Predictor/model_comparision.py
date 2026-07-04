import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("Laptop_Price_Predictor/data/cleaned_laptop_data.csv")


class LaptopPred:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X_train, Y_train):
        X_train = np.array(X_train, dtype=np.float64)
        Y_train = np.array(Y_train, dtype=np.float64)

        X_train = np.insert(X_train, 0, 1, axis=1)

        betas = np.linalg.pinv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(Y_train)

        self.coef = betas[1:]
        self.intercept = betas[0]

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = np.dot(X_test, self.coef) + self.intercept
        return y_pred


X = df.drop(columns=["Price_Euro", "Laptop", "Model"])
Y = df["Price_Euro"]
cat_cols = ["Status", "Brand", "CPU", "Storage_Type", "GPU", "Touch"]

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X["Screen"] = X["Screen"].fillna(X["Screen"].median())


# train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=56
)


scratch_model = LaptopPred()
scratch_model.fit(X_train, Y_train)
scratch_pred = scratch_model.predict(X_test)

sklearn_model = LinearRegression()
sklearn_model.fit(X_train, Y_train)
sklearn_pred = sklearn_model.predict(X_test)

print("=" * 60)
print("MULTIPLE LINEAR REGRESSION MODEL COMPARISON")
print("=" * 60)

print("\nScratch Model Results")
print("-" * 60)
print("R² Score :", r2_score(Y_test, scratch_pred))
print("MAE      :", mean_absolute_error(Y_test, scratch_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, scratch_pred)))
print("Intercept:", scratch_model.intercept)

print("\nScikit-learn Model Results")
print("-" * 60)
print("R² Score :", r2_score(Y_test, sklearn_pred))
print("MAE      :", mean_absolute_error(Y_test, sklearn_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, sklearn_pred)))
print("Intercept:", sklearn_model.intercept_)

comparison = pd.DataFrame(
    {
        "Actual": Y_test.values,
        "Scratch_Predicted": scratch_pred,
        "Sklearn_Predicted": sklearn_pred,
    }
)

print("\nActual vs Predicted")
print("-" * 60)
print(comparison.head(10))
