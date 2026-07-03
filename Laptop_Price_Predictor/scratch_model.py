import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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


# Formattiong data
X = df.drop(columns=["Price_Euro", "Laptop", "Model"])
Y = df["Price_Euro"]
cat_cols = ["Status", "Brand", "CPU", "Storage_Type", "GPU", "Touch"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X["Screen"] = X["Screen"].fillna(X["Screen"].median())
# train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=56
)

model = LaptopPred()
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)

print("R² Score :", r2_score(Y_test, Y_pred))
print("MAE      :", mean_absolute_error(Y_test, Y_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, Y_pred)))
print(f"Total Coefficients : {len(model.coef)}")
print(f"Intercept : {model.intercept:.2f}")
