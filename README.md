# 📊 End-to-End Sales Forecasting & Demand Intelligence System

## Overview

This project presents an end-to-end retail analytics solution developed using Python, Machine Learning, Time Series Forecasting, and Streamlit. The system analyzes four years of Superstore sales data to identify historical sales patterns, forecast future demand, detect unusual sales behaviour, and segment products based on demand characteristics.

The objective is to support inventory planning, demand forecasting, and business decision-making through interactive visualizations and data-driven insights.



## Key Features

- Comprehensive Exploratory Data Analysis (EDA)
- Time-series decomposition and stationarity analysis
- Sales forecasting using SARIMA, Facebook Prophet, and XGBoost
- Model comparison using MAE, RMSE, and MAPE
- Category-wise and Region-wise sales forecasting
- Weekly anomaly detection using Isolation Forest and Z-Score methods
- Product demand segmentation using K-Means Clustering
- Interactive Streamlit dashboard with business insights
- Executive business report with actionable recommendations



## Forecasting Models

Three forecasting techniques were implemented and evaluated.

### SARIMA
A statistical time-series model capable of capturing trend and seasonal behaviour in historical sales data.

### Facebook Prophet
A forecasting model designed to identify trend and yearly seasonality while handling business time-series effectively.

### XGBoost
A machine learning approach built using lag features, rolling averages, month, quarter, and seasonal information. Model performance was evaluated using MAE, RMSE, and MAPE, and the best-performing model was selected for deployment.



## Anomaly Detection

Weekly sales data was analyzed using two different anomaly detection techniques.

- Isolation Forest identified unusual sales patterns based on data distribution.
- Z-Score detection identified statistically significant deviations from normal sales behaviour.

The detected anomalies provide valuable insights into seasonal demand, promotional events, inventory shortages, and unexpected business fluctuations.



## Product Demand Segmentation

K-Means clustering was applied to group product sub-categories using:

- Total Sales
- Average Order Value
- Sales Volatility
- Growth Rate

Four demand segments were identified:

- Premium High-Value Products
- High Volume Products
- Steady Growth Products
- Emerging Demand

These demand groups help businesses optimize inventory allocation and replenishment strategies.



## Interactive Dashboard

The Streamlit dashboard includes:

- 📈 Executive KPI Dashboard
- 📊 Sales Overview and Trend Analysis
- 🔮 Forecast Explorer
- ⚠️ Sales Anomaly Report
- 📦 Product Demand Segmentation
- 📥 Downloadable Reports



## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Statsmodels
- Facebook Prophet
- XGBoost
- Streamlit



## Project Structure

```
SalesForecasting-Demand-Intelligence-System/
│
├── app.py
├── train.csv
├── analysis.ipynb
├── summary.docx
├── requirements.txt
└── README.md
```

---

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```



## Author

**Chirra Nandhini Reddy**

B.Tech – Computer Science & Data Science

Hyderabad Institute of Technology and Management



## Conclusion

This project demonstrates a complete retail analytics pipeline by integrating exploratory data analysis, forecasting, anomaly detection, demand segmentation, and interactive visualization into a single decision-support system. The solution enables organizations to make informed inventory and supply chain decisions based on historical trends and predictive analytics.
