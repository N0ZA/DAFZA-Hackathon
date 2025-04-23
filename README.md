# AI-Powered F&B Demand Forecasting for UAE Hotels

## 🚀 Project Overview

This project aims to **predict Food & Beverage (F&B) demand** for hotels in the UAE using machine learning (ML) techniques. By forecasting demand more accurately, we aim to optimize procurement, reduce waste, and enhance overall operational efficiency in hotel kitchens and restaurants. This model leverages historical data from products, operational data, and weekly demand to create reliable predictions for perishable goods.

---

## 🧾 Use Case

The UAE's hospitality industry faces significant challenges in forecasting F&B demand due to fluctuations in guest behavior driven by **holidays, seasons**, and **global climate factors**. By applying **AI and ML** to the problem, this project predicts demand for key F&B items, leading to:

- **Reduced procurement costs**
- **Optimized resource allocation**
- **Minimized waste and spoilage**

---

## 📦 Features

- **Data Collection**: Gathers data from 3 CSV files: `products.csv`, `weekly_operational_data.csv`, and `weekly_demand.csv`.
- **Feature Engineering**: Converts dates to Hijri calendar features, handles categorical variables, and adds seasonal factors like `temperature` and `guests`.
- **Forecasting Models**: Implements both **RandomForestRegressor** and **LightGBM** for accurate demand prediction.
- **Visualization**: Generates 3 key graphs:
  1. Product-level demand trends
  2. Weekly operational data (guests)
  3. Comparison of predicted vs. actual weekly demand

---

## 🛠️ Tech Stack

- **Python** (pandas, numpy, scikit-learn, lightgbm, hijri-converter)
- **Matplotlib** and **Seaborn** for visualizations
- **Jupyter Notebooks** for exploration and experimentation
- **Scikit-learn** for model training, splitting, and evaluation

---

.
## 📈 Model Evaluation

The model is evaluated using the following metrics:

- **MAE (Mean Absolute Error)**: Measures the average magnitude of the errors in the predictions, without considering their direction. It’s the sum of the absolute errors divided by the number of predictions.
  
- **RMSE (Root Mean Squared Error)**: A widely used metric that penalizes large errors more heavily than smaller ones, calculated by taking the square root of the average of the squared differences between predicted and actual values.

- **MAPE (Mean Absolute Percentage Error)**: Represents the percentage difference between predicted and actual values. It’s calculated as the average of the absolute percentage differences between actual and predicted values.







