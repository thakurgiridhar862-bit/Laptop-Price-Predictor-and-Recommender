# 💻 Laptop Recommendation and Price Prediction System

A Machine Learning project that predicts laptop prices using **Multiple Linear Regression** and recommends laptops based on user requirements such as budget, RAM, storage, brand, and touchscreen preference.

---

## 📌 Project Overview

This project is divided into two main modules:

- **Laptop Price Prediction**
- **Laptop Recommendation System**

The prediction model is trained using Multiple Linear Regression, while the recommendation system uses rule-based filtering to suggest the most suitable laptops.

---

## 🚀 Features

### Laptop Price Prediction

- Predict laptop price using Multiple Linear Regression
- Real-time prediction through Streamlit
- Price shown in Euro and Indian Rupees
- Model trained using Scikit-learn
- Model saved using Joblib

### Laptop Recommendation

- Budget Based Recommendation
- RAM Based Filtering
- Storage Based Filtering
- Brand Based Filtering
- Touch Screen Filtering
- Top 5 Laptop Recommendations

### Model Comparison

- Multiple Linear Regression using Scikit-learn
- Multiple Linear Regression implemented from Scratch
- Performance comparison of both models

---

## 📊 Model Performance

| Metric | Value |
|---------|-------|
| R² Score | 0.8205 |
| MAE | 259.77 |
| RMSE | 378.39 |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Structure

```text
Laptop_Price_Predictor/

│── data/
│── graphs/
│── models/
│── screenshots/

│── analysis.py
│── model_training.py
│── scratch_model.py
│── model_comparison.py
│── recommendation.py
│── app.py

│── requirements.txt
│── README.md
```

---

## 📸 Screenshots

### Home Page

```
screenshots/Dashboard.png
```

### Price Prediction

```
screenshots/Final_Dashboard1.png
```

### Laptop Recommendation

```
screenshots/Rec_Dashboard.png
```

---

## ▶️ Installation

Clone the repository

```bash
git clone <repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
python -m streamlit run Laptop_Price_Predictor/app.py
```

---

## 📈 Workflow

```
Dataset

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Encoding

↓

Train Test Split

↓

Multiple Linear Regression

↓

Model Evaluation

↓

Model Saving

↓

Laptop Recommendation

↓

Streamlit Application
```

---

## 🔮 Future Improvements

- Random Forest Regressor
- XGBoost Regressor
- Hyperparameter Tuning
- Better Recommendation Ranking
- Improved User Interface

---

## Live Demo

https://laptop-price-predictor-and-recommender.streamlit.app/

## 👨‍💻 Developer

**Giridhar Jadon**