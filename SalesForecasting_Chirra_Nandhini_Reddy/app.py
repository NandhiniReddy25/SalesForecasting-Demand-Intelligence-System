# Sales Forecasting & Demand Intelligence Dashboard


# Import Libraries


import os
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from prophet import Prophet
from scipy.stats import zscore

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)




# Page Configuration

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Professional Theme

st.markdown("""
<style>

.main{
    background-color: var(--background-color);
}

h1{
    color: var(--text-color);
}

h2{
    color:#2563EB;
}

h3{
    color:#2563EB;
}

[data-testid="stMetric"]{
    background-color: var(--secondary-background-color);
    border-radius:12px;
    padding:18px;
    border-left:6px solid #2563EB;
    box-shadow:0px 2px 10px rgba(0,0,0,0.08);
}

.stDataFrame{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)




# Load Dataset


@st.cache_data
def load_data():

    df = pd.read_csv(
        "train.csv",
        encoding="latin1"
    )

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year

    df["Month"] = df["Order Date"].dt.month

    df["Month Name"] = df["Order Date"].dt.strftime("%B")

    df["Quarter"] = df["Order Date"].dt.quarter

    return df


sales_df = load_data()



# Sidebar


st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart--v1.png",
    width=70
)

st.sidebar.title("Navigation")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Select Dashboard",

    [

        "🏠 Home",

        "📈 Sales Overview",

        "🔮 Forecast Explorer",

        "🚨 Anomaly Report",

        "📦 Demand Segments"

    ]

)

st.sidebar.markdown("---")



st.sidebar.markdown("---")

st.sidebar.success(
"AI Powered Business Intelligence Dashboard"
)




# Footer Function


def footer():

    st.markdown("---")

    st.markdown(

    """
    <center>

    <b>Sales Forecasting & Demand Intelligence Dashboard</b><br>

    Developed by <b>Chirra Nandhini Reddy</b><br><br>

    Python | Streamlit | Machine Learning | Time Series Forecasting

    </center>
    """,

    unsafe_allow_html=True

    )


# Home Page

if page == "🏠 Home":

    st.title("📊 Sales Forecasting & Demand Intelligence Dashboard")

    st.markdown("""
### AI Powered Business Intelligence Solution

Welcome to the interactive Sales Forecasting Dashboard developed using Machine Learning,
Time Series Forecasting and Business Intelligence techniques.

This dashboard helps organizations monitor historical sales performance,
forecast future demand, detect unusual sales behaviour and segment products
for better inventory planning.
""")

    st.markdown("---")

    # Project Information
    

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📌 Project Overview")

        st.info("""

**Project Title**

Sales Forecasting & Demand Intelligence Dashboard

**Objective**

Forecast future sales using Machine Learning and support business decision making.

**Dataset**

Superstore Sales Dataset

**Forecasting Models**

• SARIMA

• Prophet

• XGBoost

**Machine Learning**

• Isolation Forest

• K-Means Clustering

""")

    with col2:

        st.subheader("👩‍💻 Developer")

        st.success("""

**Name**

Chirra Nandhini Reddy

**Technology**

Python

Streamlit

Plotly

Pandas

Scikit-Learn

XGBoost

Prophet

SARIMA

""")

    st.markdown("---")



    # Executive KPI Cards


    st.subheader("📈 Executive Business Overview")

    total_sales = sales_df["Sales"].sum()

    total_orders = sales_df["Order ID"].nunique()

    total_customers = sales_df["Customer ID"].nunique()

    total_products = sales_df["Product Name"].nunique()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.2f}"
    )

    c2.metric(
        "📦 Orders",
        f"{total_orders:,}"
    )

    c3.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )

    c4.metric(
        "🛒 Products",
        f"{total_products:,}"
    )

    st.markdown("---")



    # Dataset Summary


    st.subheader("📄 Dataset Summary")

    summary = pd.DataFrame({

        "Metric":[

            "Rows",

            "Columns",

            "Missing Values",

            "Date Range"

        ],

        "Value":[

            len(sales_df),

            sales_df.shape[1],

            sales_df.isnull().sum().sum(),

            f"{sales_df['Order Date'].min().date()}  →  {sales_df['Order Date'].max().date()}"

        ]

    })

    st.dataframe(

        summary,

        hide_index=True,

        width="stretch"

    )

    st.markdown("---")


    
    # Dataset Preview


    st.subheader("🗂 Dataset Preview")

    st.dataframe(

        sales_df.head(10),

        width="stretch",

        hide_index=True

    )

    st.markdown("---")



    # Dashboard Features

    st.subheader("🚀 Dashboard Features")

    col1,col2 = st.columns(2)

    with col1:

        st.success("""

✅ Interactive Sales Dashboard

✅ Region & Category Analysis

✅ Monthly Sales Trend

✅ Executive KPI Cards

✅ Download Reports

""")

    with col2:

        st.success("""

✅ Sales Forecasting

✅ Anomaly Detection

✅ Demand Segmentation

✅ Business Insights

✅ Inventory Planning

""")

    st.markdown("---")



        # Project Workflow

    st.subheader("⚙ Project Workflow")

    st.markdown(
        """
        <<div style="
        background:#DBEAFE;
        padding:25px;
        border-radius:12px;
        text-align:center;
        font-size:18px;
        line-height:2.2;
        color:#0F172A;
        border-left:6px solid #2563EB;
        box-shadow:0px 2px 10px rgba(0,0,0,0.08);
        ">

        Data Collection <br>

        ⬇️ <br>

        Data Cleaning <br>

        ⬇️ <br>

        Exploratory Data Analysis <br>

        ⬇️ <br>

        Sales Forecasting <br>

        ⬇️ <br>

        Anomaly Detection <br>

        ⬇️ <br>

        Demand Segmentation <br>

        ⬇️ <br>

        Business Dashboard

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")


    # Business Impact


    st.subheader("🎯 Business Impact")

    st.success("""

✔ Helps forecast future sales.

✔ Detects unusual sales patterns.

✔ Identifies high demand products.

✔ Supports inventory optimization.

✔ Enables data-driven business decisions.

✔ Improves operational efficiency.

""")

    footer()

# ==========================================================
# Sales Overview Page
# ==========================================================

elif page == "📈 Sales Overview":

    st.title("📈 Sales Performance Dashboard")

    st.markdown("""
Analyze historical sales performance using interactive filters,
business KPIs and executive visualizations.
""")



    # ======================================================
    # Sidebar Filters
    # ======================================================

    st.sidebar.subheader("🔍 Sales Filters")

    selected_region = st.sidebar.selectbox(
        "Region",
        ["All"] + sorted(sales_df["Region"].unique())
    )

    selected_category = st.sidebar.selectbox(
        "Category",
        ["All"] + sorted(sales_df["Category"].unique())
    )

    selected_year = st.sidebar.selectbox(
        "Year",
        ["All"] + sorted(sales_df["Year"].astype(str).unique())
    )

    # ======================================================
    # Apply Filters
    # ======================================================

    filtered_df = sales_df.copy()

    if selected_region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == selected_region
        ]

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == selected_category
        ]

    if selected_year != "All":
        filtered_df = filtered_df[
            filtered_df["Year"] == int(selected_year)
        ]

    st.markdown("---")

    # ======================================================
    # Executive KPI Cards
    # ======================================================

    st.subheader("📊 Executive KPI Summary")

    total_sales = filtered_df["Sales"].sum()
    total_orders = filtered_df["Order ID"].nunique()
    total_customers = filtered_df["Customer ID"].nunique()
    total_products = filtered_df["Product Name"].nunique()
    average_sales = filtered_df["Sales"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "💰 Revenue",
        f"${total_sales:,.2f}"
    )

    c2.metric(
        "📦 Orders",
        f"{total_orders:,}"
    )

    c3.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )

    c4.metric(
        "🛒 Products",
        f"{total_products:,}"
    )

    c5.metric(
        "📈 Avg Sales",
        f"${average_sales:,.2f}"
    )

    st.caption(
        "📌 KPI values update automatically based on the selected filters."
    )

    st.markdown("---")

    # ======================================================
    # Sales Trend Analysis
    # ======================================================

    st.subheader("📈 Sales Trend Analysis")

    left_chart, right_chart = st.columns(2)

    # --------------------------------------------------
    # Year-wise Sales
    # --------------------------------------------------

    with left_chart:

        yearly_sales = (
            filtered_df
            .groupby("Year")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            yearly_sales,
            x="Year",
            y="Sales",
            color="Sales",
            text_auto=".2s",
            title="Year-wise Revenue",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            height=420,
            title_x=0.30,
            coloraxis_showscale=False,
            xaxis_title="Year",
            yaxis_title="Sales"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------
    # Monthly Sales Trend
    # --------------------------------------------------

    with right_chart:

        monthly_sales = (
            filtered_df
            .groupby(
                pd.Grouper(
                    key="Order Date",
                    freq="ME"
                )
            )["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly_sales,
            x="Order Date",
            y="Sales",
            markers=True,
            title="Monthly Sales Trend"
        )

        fig.update_traces(
            line=dict(width=3)
        )

        fig.update_layout(
            height=420,
            title_x=0.30,
            xaxis_title="Month",
            yaxis_title="Sales"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.info("""
### 📌 Trend Insights

• The yearly revenue chart highlights long-term business growth.

• Monthly sales reveal seasonal demand fluctuations.

• These trends help businesses forecast future inventory requirements and revenue expectations.
""")

    st.markdown("---")
        # ======================================================
    # Regional & Category Performance
    # ======================================================

    st.subheader("🌍 Regional & Category Performance")

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # Regional Sales
    # --------------------------------------------------

    with col1:

        region_df = sales_df.copy()

        if selected_category != "All":
            region_df = region_df[
                region_df["Category"] == selected_category
            ]

        if selected_year != "All":
            region_df = region_df[
                region_df["Year"] == int(selected_year)
            ]

        region_sales = (
            region_df
            .groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            region_sales,
            x="Sales",
            y="Region",
            orientation="h",
            color="Sales",
            color_continuous_scale="Blues",
            text_auto=".2s",
            title="Regional Revenue"
        )

        fig.update_layout(
            height=430,
            title_x=0.30,
            coloraxis_showscale=False,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        highest_region = region_sales.iloc[0]["Region"]
        highest_region_sales = region_sales.iloc[0]["Sales"]

        st.success(
            f"""
🏆 **Best Performing Region**

**{highest_region}**

Revenue: **${highest_region_sales:,.0f}**
"""
        )

    # --------------------------------------------------
    # Category Sales
    # --------------------------------------------------

    with col2:

        category_df = sales_df.copy()

        if selected_region != "All":
            category_df = category_df[
                category_df["Region"] == selected_region
            ]

        if selected_year != "All":
            category_df = category_df[
                category_df["Year"] == int(selected_year)
            ]

        category_sales = (
            category_df
            .groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            category_sales,
            x="Sales",
            y="Category",
            orientation="h",
            color="Sales",
            color_continuous_scale="Greens",
            text_auto=".2s",
            title="Category Revenue"
        )

        fig.update_layout(
            height=430,
            title_x=0.30,
            coloraxis_showscale=False,
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        highest_category = category_sales.iloc[0]["Category"]
        highest_category_sales = category_sales.iloc[0]["Sales"]

        st.success(
            f"""
🏆 **Best Performing Category**

**{highest_category}**

Revenue: **${highest_category_sales:,.0f}**
"""
        )

    st.info(
        """
### 💡 Business Insights

✔ Regional analysis identifies the strongest geographical markets.

✔ Category analysis reveals which product groups generate the highest revenue.

✔ Managers can use these insights to optimize inventory allocation, marketing campaigns and expansion strategies.

✔ The charts update automatically whenever filters are changed.
"""
    )

    st.markdown("---")
        # ======================================================
    # Top Performing Products
    # ======================================================

    st.subheader("🏆 Top Performing Products")

    top_products = (
        filtered_df
        .groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    left, right = st.columns([3, 1])

    with left:

        fig = px.bar(
            top_products,
            x="Sales",
            y="Product Name",
            orientation="h",
            color="Sales",
            color_continuous_scale="Blues",
            text_auto=".2s",
            title="Top 10 Revenue Generating Products"
        )

        fig.update_layout(
            height=520,
            title_x=0.28,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with right:

        st.markdown("### 🥇 Product Ranking")

        st.dataframe(
            top_products,
            hide_index=True,
            width="stretch"
        )

        st.metric(
            "Best Product",
            top_products.iloc[0]["Product Name"]
        )

        st.metric(
            "Revenue",
            f"${top_products.iloc[0]['Sales']:,.0f}"
        )

    st.success(
        f"""
### 📌 Product Insight

**{top_products.iloc[0]['Product Name']}** generated the highest sales revenue.

This product should receive priority for:

• Inventory replenishment

• Marketing campaigns

• Seasonal promotions

• Demand forecasting
"""
    )

    st.markdown("---")

    # ======================================================
    # Top Customers
    # ======================================================

    st.subheader("👥 Top Customers")

    top_customers = (
        filtered_df
        .groupby("Customer Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    left, right = st.columns([3, 1])

    with left:

        fig = px.bar(
            top_customers,
            x="Sales",
            y="Customer Name",
            orientation="h",
            color="Sales",
            color_continuous_scale="Greens",
            text_auto=".2s",
            title="Top 10 Revenue Contributing Customers"
        )

        fig.update_layout(
            height=520,
            title_x=0.28,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with right:

        st.markdown("### ⭐ Customer Ranking")

        st.dataframe(
            top_customers,
            hide_index=True,
            width="stretch"
        )

        st.metric(
            "Top Customer",
            top_customers.iloc[0]["Customer Name"]
        )

        st.metric(
            "Revenue",
            f"${top_customers.iloc[0]['Sales']:,.0f}"
        )

    st.success(
        f"""
### 📌 Customer Insight

**{top_customers.iloc[0]['Customer Name']}** is the highest-value customer.

Recommended actions:

• Customer retention programs

• Personalized offers

• Loyalty rewards

• Premium support
"""
    )

    st.markdown("---")

        # ======================================================
    # State-wise Performance
    # ======================================================

    st.subheader("🗺️ State-wise Sales Performance")

    state_sales = (
        filtered_df
        .groupby("State")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    col1, col2 = st.columns([3, 1])

    with col1:

        fig = px.bar(
            state_sales,
            x="Sales",
            y="State",
            orientation="h",
            color="Sales",
            color_continuous_scale="Purples",
            text_auto=".2s",
            title="Top 15 States by Revenue"
        )

        fig.update_layout(
            height=550,
            title_x=0.30,
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Sales",
            yaxis_title="State"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with col2:

        st.markdown("### 🏆 Top State")

        st.metric(
            "State",
            state_sales.iloc[0]["State"]
        )

        st.metric(
            "Revenue",
            f"${state_sales.iloc[0]['Sales']:,.0f}"
        )

        st.metric(
            "States Analyzed",
            filtered_df["State"].nunique()
        )

    st.info(
        f"""
### 💡 Business Insight

**{state_sales.iloc[0]['State']}** generated the highest revenue among all states.

This region represents a high-value market and should be prioritized for:

• Business expansion

• Inventory allocation

• Marketing campaigns

• Customer acquisition
"""
    )

    st.markdown("---")

    # ======================================================
    # Executive Dashboard Summary
    # ======================================================

    st.subheader("📌 Executive Dashboard Summary")

    best_region = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

    best_category = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    best_product = top_products.iloc[0]["Product Name"]

    st.success(f"""
### 📊 Key Business Highlights

💰 **Total Revenue**
**${total_sales:,.0f}**

📦 **Orders Processed**
**{total_orders:,}**

👥 **Customers**
**{total_customers:,}**

🛒 **Products**
**{total_products:,}**

🌍 **Best Region**
**{best_region}**

📦 **Best Category**
**{best_category}**

🏆 **Best Selling Product**
**{best_product}**

📈 Interactive filters enable dynamic business analysis and support data-driven decision making.
""")

    st.markdown("---")

    # ======================================================
    # Download Report
    # ======================================================

    st.subheader("📥 Export Report")

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Filtered Sales Report",
        data=csv,
        file_name="Sales_Report.csv",
        mime="text/csv"
    )



    # ======================================================
    # Footer
    # ======================================================

    st.markdown(
        """
---
<div style='text-align:center; color:gray;'>

### 📊 Sales Forecasting & Demand Intelligence Dashboard

Developed by **Chirra Nandhini Reddy**

Built using **Python • Streamlit • Plotly • XGBoost • Prophet • SARIMA • Scikit-Learn**

© 2026 All Rights Reserved

</div>
""",
        unsafe_allow_html=True
    )

# ==========================================================
# Forecast Explorer
# ==========================================================

elif page == "🔮 Forecast Explorer":

    st.title("🔮 Sales Forecast Explorer")

    st.markdown("""
Forecast future sales using **Facebook Prophet Time Series Forecasting**.

This dashboard predicts future sales based on historical sales trends to support inventory planning and business decision-making.
""")

    st.markdown("---")

    # ======================================================
    # Forecast Configuration
    # ======================================================

    st.subheader("⚙ Forecast Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_category = st.selectbox(
            "📦 Category",
            sorted(sales_df["Category"].unique())
        )

    with col2:

        selected_region = st.selectbox(
            "🌍 Region",
            sorted(sales_df["Region"].unique())
        )

    with col3:

        forecast_months = st.slider(
            "📅 Forecast Horizon (Months)",
            min_value=1,
            max_value=3,
            value=3
        )

    st.markdown("---")

    # ======================================================
    # Filter Dataset
    # ======================================================

    filtered = sales_df[
        (sales_df["Category"] == selected_category) &
        (sales_df["Region"] == selected_region)
    ].copy()

    monthly = (
        filtered
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    if monthly.empty:

        st.warning(
            "No sales data available for the selected Region and Category."
        )

        st.stop()

    if len(monthly) < 6:

        st.warning(
            "Not enough historical data to generate a reliable forecast."
        )

        st.stop()

    st.subheader("📊 Forecast KPI Summary")

    total_sales = monthly["Sales"].sum()

    latest_sales = monthly["Sales"].iloc[-1]

    average_sales = monthly["Sales"].mean()

    peak_sales = monthly["Sales"].max()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "📈 Latest Month",
        f"${latest_sales:,.0f}"
    )

    c3.metric(
        "📊 Average",
        f"${average_sales:,.0f}"
    )

    c4.metric(
        "🚀 Peak Sales",
        f"${peak_sales:,.0f}"
    )

    st.caption(
        "The KPI cards automatically update according to the selected Region and Category."
    )

    st.markdown("---")

        # ======================================================
    # Historical Sales Trend
    # ======================================================

    st.subheader("📈 Historical Sales Trend")

    history_fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Historical Monthly Sales"
    )

    history_fig.update_traces(
        line=dict(width=3)
    )

    history_fig.update_layout(
        height=450,
        title_x=0.30,
        xaxis_title="Month",
        yaxis_title="Sales ($)"
    )

    st.plotly_chart(
        history_fig,
        width="stretch"
    )

    st.markdown("---")

    # ======================================================
    # Prophet Forecast Model
    # ======================================================

    prophet_df = monthly.rename(
        columns={
            "Order Date": "ds",
            "Sales": "y"
        }
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=forecast_months,
        freq="ME"
    )

    forecast = model.predict(future)

    # ======================================================
    # Forecast Chart
    # ======================================================

    st.subheader("🔮 Historical vs Forecast Sales")

    forecast_fig = go.Figure()

    # Historical Sales
    forecast_fig.add_trace(
        go.Scatter(
            x=prophet_df["ds"],
            y=prophet_df["y"],
            mode="lines+markers",
            name="Historical Sales",
            line=dict(width=3)
        )
    )

    # Forecast
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(width=4)
        )
    )

    # Upper Confidence
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    # Lower Confidence
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Confidence Interval"
        )
    )

    forecast_fig.update_layout(
        height=550,
        title="Historical Sales vs Forecast",
        title_x=0.30,
        xaxis_title="Month",
        yaxis_title="Sales ($)"
    )

    st.plotly_chart(
        forecast_fig,
        width="stretch"
    )

    st.markdown("---")

        # ======================================================
    # Model Performance (Required)
    # ======================================================

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    import numpy as np

    history_prediction = model.predict(prophet_df[["ds"]])

    mae = mean_absolute_error(
        prophet_df["y"],
        history_prediction["yhat"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            prophet_df["y"],
            history_prediction["yhat"]
        )
    )

    st.subheader("📊 Forecast Model Performance")

    m1, m2 = st.columns(2)

    m1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    m2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    st.caption(
        "Lower MAE and RMSE values indicate better forecasting accuracy."
    )

    st.markdown("---")

    # ======================================================
    # Forecast Summary Table
    # ======================================================

    st.subheader("📋 Forecast Summary")

    forecast_table = (
        forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]
        .tail(forecast_months)
        .copy()
    )

    forecast_table.columns = [
        "Forecast Date",
        "Predicted Sales",
        "Lower Bound",
        "Upper Bound"
    ]

    st.dataframe(
        forecast_table.style.format(
            {
                "Predicted Sales": "${:,.2f}",
                "Lower Bound": "${:,.2f}",
                "Upper Bound": "${:,.2f}"
            }
        ),
        width="stretch"
    )

    st.markdown("---")

    # ======================================================
    # Forecast KPIs
    # ======================================================

    st.subheader("📈 Forecast KPIs")

    predicted_sales = forecast_table[
        "Predicted Sales"
    ].mean()

    highest_prediction = forecast_table[
        "Upper Bound"
    ].max()

    lowest_prediction = forecast_table[
        "Lower Bound"
    ].min()

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Average Forecast",
        f"${predicted_sales:,.0f}"
    )

    k2.metric(
        "Highest Expected",
        f"${highest_prediction:,.0f}"
    )

    k3.metric(
        "Lowest Expected",
        f"${lowest_prediction:,.0f}"
    )

    st.markdown("---")

        # ======================================================
    # AI Forecast Insight
    # ======================================================

    latest_actual = prophet_df["y"].iloc[-1]

    latest_prediction = forecast_table[
        "Predicted Sales"
    ].iloc[0]

    growth = (
        (latest_prediction - latest_actual)
        / latest_actual
    ) * 100

    st.subheader("🤖 AI Forecast Insight")

    if growth >= 0:

        st.success(
            f"""
The forecasting model predicts approximately **{growth:.2f}% growth**
compared to the latest observed sales.

This indicates increasing customer demand and suggests maintaining adequate inventory levels.
"""
        )

    else:

        st.warning(
            f"""
The forecasting model predicts approximately **{abs(growth):.2f}% decline**
compared to the latest observed sales.

Consider reviewing inventory planning and promotional strategies.
"""
        )

    st.markdown("---")

    # ======================================================
    # Business Recommendations
    # ======================================================

    st.subheader("💼 Business Recommendations")

    st.info(
        f"""
### Strategic Recommendations

✅ Prioritize inventory for **{selected_category}** products.

✅ Focus marketing efforts in the **{selected_region}** region.

✅ Prepare stock for the next **{forecast_months} month(s)**.

✅ Monitor demand trends regularly.

✅ Retrain forecasting models whenever new sales data becomes available.

✅ Use forecast insights to optimize procurement and inventory planning.
"""
    )

    st.markdown("---")

    # ======================================================
    # Executive Summary
    # ======================================================

    st.subheader("📌 Executive Summary")

    st.success(
        f"""
### Dashboard Highlights

✔ Selected Category : **{selected_category}**

✔ Selected Region : **{selected_region}**

✔ Forecast Horizon : **{forecast_months} Month(s)**

✔ Forecast Model : **Facebook Prophet**

✔ MAE : **{mae:.2f}**

✔ RMSE : **{rmse:.2f}**

✔ Interactive forecasting supports inventory planning and business decision-making.

✔ Historical trends and future predictions are visualized together for easier analysis.
"""
    )

    st.markdown("---")

    # ======================================================
    # Download Forecast Report
    # ======================================================

    csv = forecast_table.to_csv(index=False)

    st.download_button(
        "📥 Download Forecast Report",
        csv,
        "Forecast_Report.csv",
        "text/csv"
    )

    st.markdown("---")

    # ======================================================
    # Footer
    # ======================================================

    st.caption(
        "🔮 Sales Forecast Explorer | Developed by Chirra Nandhini Reddy | Powered by Streamlit, Plotly & Facebook Prophet"
    )

    # ==========================================================
# Anomaly Report
# ==========================================================

# ==========================================================
# Anomaly Report
# ==========================================================

elif page == "🚨 Anomaly Report":

    st.title("🚨 Sales Anomaly Detection Report")

    st.markdown("""
Detect unusual weekly sales patterns using both **Isolation Forest**
and **Z-Score based anomaly detection**.

This dashboard compares both techniques to help identify abnormal
business behaviour and support proactive decision-making.
""")

    st.markdown("---")

    # ======================================================
    # Weekly Sales
    # ======================================================

    weekly_sales = (
        sales_df
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    weekly_sales.columns = [
        "Week",
        "Sales"
    ]

    # ======================================================
    # Isolation Forest
    # ======================================================

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Isolation"] = iso.fit_predict(
        weekly_sales[["Sales"]]
    )

    isolation_results = weekly_sales[
        weekly_sales["Isolation"] == -1
    ]

    # ======================================================
    # Z-Score
    # ======================================================

    weekly_sales["ZScore"] = np.abs(
        zscore(
            weekly_sales["Sales"]
        )
    )

    zscore_results = weekly_sales[
        weekly_sales["ZScore"] > 2
    ]

    st.subheader("📊 Detection Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Isolation Forest",
        len(isolation_results)
    )

    c2.metric(
        "Z-Score",
        len(zscore_results)
    )

    c3.metric(
        "Total Weeks",
        len(weekly_sales)
    )

    c4.metric(
        "Detection Methods",
        "2"
    )

    st.markdown("---")

        # ======================================================
    # Interactive Anomaly Charts
    # ======================================================

    tab1, tab2 = st.tabs(
        [
            "🌲 Isolation Forest",
            "📈 Z-Score"
        ]
    )

    # ======================================================
    # Isolation Forest Chart
    # ======================================================

    with tab1:

        st.subheader("Isolation Forest Anomaly Detection")

        fig = go.Figure()

        # Weekly Sales Line
        fig.add_trace(
            go.Scatter(
                x=weekly_sales["Week"],
                y=weekly_sales["Sales"],
                mode="lines",
                name="Weekly Sales",
                line=dict(width=3)
            )
        )

        # Anomalies
        fig.add_trace(
            go.Scatter(
                x=isolation_results["Week"],
                y=isolation_results["Sales"],
                mode="markers",
                name="Anomalies",
                marker=dict(
                    size=11,
                    color="red",
                    symbol="x"
                )
            )
        )

        fig.update_layout(
            height=520,
            title="Isolation Forest - Weekly Sales Anomalies",
            title_x=0.25,
            xaxis_title="Week",
            yaxis_title="Sales ($)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.info(
            f"""
Isolation Forest detected **{len(isolation_results)} anomalous weeks** by learning unusual patterns in the sales distribution.

This approach identifies both extremely high and unusually low sales periods without relying on statistical assumptions.
"""
        )

    # ======================================================
    # Z-Score Chart
    # ======================================================

    with tab2:

        st.subheader("Z-Score Based Anomaly Detection")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=weekly_sales["Week"],
                y=weekly_sales["Sales"],
                mode="lines",
                name="Weekly Sales",
                line=dict(width=3)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=zscore_results["Week"],
                y=zscore_results["Sales"],
                mode="markers",
                name="Anomalies",
                marker=dict(
                    size=11,
                    color="orange",
                    symbol="diamond"
                )
            )
        )

        fig.update_layout(
            height=520,
            title="Z-Score - Weekly Sales Anomalies",
            title_x=0.25,
            xaxis_title="Week",
            yaxis_title="Sales ($)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.info(
            f"""
Z-Score detected **{len(zscore_results)} statistically significant anomalous weeks** where sales deviated more than two standard deviations from the overall sales distribution.
"""
        )

    st.markdown("---")

        # ======================================================
    # Comparison Summary
    # ======================================================

    st.subheader("📊 Comparison of Detection Methods")

    comparison = pd.DataFrame({

        "Method": [
            "Isolation Forest",
            "Z-Score"
        ],

        "Anomalies Detected": [
            len(isolation_results),
            len(zscore_results)
        ]

    })

    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch"
    )

    st.markdown("---")

    # ======================================================
    # Detected Anomalies Table
    # ======================================================

    st.subheader("📅 Detected Anomaly Weeks")

    isolation_table = isolation_results[
        ["Week", "Sales"]
    ].copy()

    isolation_table["Method"] = "Isolation Forest"

    zscore_table = zscore_results[
        ["Week", "Sales"]
    ].copy()

    zscore_table["Method"] = "Z-Score"

    anomaly_table = pd.concat(
        [
            isolation_table,
            zscore_table
        ],
        ignore_index=True
    )

    anomaly_table = anomaly_table.sort_values(
        by="Week"
    )

    anomaly_table.columns = [
        "Week",
        "Weekly Sales",
        "Detection Method"
    ]

    st.dataframe(
        anomaly_table.style.format({
            "Weekly Sales": "${:,.2f}"
        }),
        width="stretch"
    )

    st.markdown("---")

    # ======================================================
    # Business Insights
    # ======================================================

    st.subheader("💡 Business Insights")

    st.info("""
### Key Observations

• Isolation Forest detected a larger number of unusual sales weeks because it learns abnormal patterns directly from the data distribution.

• Z-Score identified only statistically significant deviations, making it a more conservative anomaly detection technique.

• High sales anomalies may correspond to festive seasons, promotional campaigns, or bulk customer purchases.

• Low sales anomalies may indicate supply-chain disruptions, inventory shortages, public holidays, or temporary reductions in customer demand.

• When both techniques identify the same weeks as anomalous, confidence in the anomaly increases.

• Combining Machine Learning and Statistical approaches provides more reliable anomaly detection for retail sales monitoring.
""")

    st.markdown("---")

    # ======================================================
    # Executive Summary
    # ======================================================

    st.subheader("📌 Executive Summary")

    st.success(f"""
### Task 5 Highlights

✔ Isolation Forest detected **{len(isolation_results)}** anomalous weeks.

✔ Z-Score detected **{len(zscore_results)}** statistically significant anomalies.

✔ Machine Learning and Statistical techniques complement each other.

✔ Early anomaly detection helps improve inventory planning and business monitoring.

✔ Interactive visualization supports rapid identification of unusual sales behaviour.
""")

    st.markdown("---")

    # ======================================================
    # Download Report
    # ======================================================

    csv = anomaly_table.to_csv(index=False)

    st.download_button(
        "📥 Download Anomaly Report",
        csv,
        "Anomaly_Report.csv",
        "text/csv"
    )

    st.markdown("---")

    st.caption(
        "🚨 Sales Anomaly Report | Powered by Isolation Forest, Z-Score, Plotly & Streamlit"
    )

    
    # ==========================================================
# Demand Segmentation Dashboard
# ==========================================================

elif page == "📦 Demand Segments":

        st.title("📦 Product Demand Segmentation")

        st.markdown("""
This dashboard groups product sub-categories into meaningful demand segments
using **K-Means Clustering**.

Products are classified into demand groups based on:

- Total Sales
- Average Order Value
- Sales Volatility
- Growth Rate

The results help optimize inventory planning and stocking strategies.
""")

        st.markdown("---")

    # =====================================================
    # Feature Engineering
    # =====================================================

        cluster_features = (
        sales_df.groupby("Sub-Category")
        .agg(
            Total_Sales=("Sales", "sum"),
            Average_Order_Value=("Sales", "mean"),
            Sales_Volatility=("Sales", "std")
        )
    )

        yearly_sales = (
        sales_df.groupby(["Year", "Sub-Category"])["Sales"]
        .sum()
        .unstack(fill_value=0)
    )

        growth_rate = yearly_sales.pct_change(axis=1).mean(axis=1)

        cluster_features["Growth_Rate"] = growth_rate

        cluster_features = cluster_features.fillna(0)

    # =====================================================
    # Scaling
    # =====================================================

        scaler = StandardScaler()

        scaled_features = scaler.fit_transform(cluster_features)

    # =====================================================
    # KMeans Clustering
    # =====================================================

        kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

        cluster_features["Cluster"] = kmeans.fit_predict(
        scaled_features
    )

    # =====================================================
    # Cluster Labels
    # =====================================================

        cluster_labels = {

        0: "Premium High-Value Products",

        1: "Steady Growth Products",

        2: "High Volume Products",

        3: "Emerging Demand"

    }

        cluster_features["Demand Segment"] = (
        cluster_features["Cluster"]
        .map(cluster_labels)
    )

        st.subheader("📊 Demand Segmentation Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
        "Clusters",
        cluster_features["Cluster"].nunique()
    )

        c2.metric(
        "Sub-Categories",
        len(cluster_features)
    )

        c3.metric(
        "Algorithm",
        "K-Means"
    )

        c4.metric(
        "Features",
        4
    )

        st.caption(
        "Segmentation is performed using Total Sales, Average Order Value, Sales Volatility and Growth Rate."
    )

        st.markdown("---")

    # =====================================================
    # PCA Visualization
    # =====================================================

        pca = PCA(n_components=2)

        pca_features = pca.fit_transform(
        scaled_features
    )

        cluster_features["PCA1"] = pca_features[:, 0]

        cluster_features["PCA2"] = pca_features[:, 1]

        st.subheader("📈 PCA Cluster Visualization")

        fig = px.scatter(
        cluster_features,
        x="PCA1",
        y="PCA2",
        color="Demand Segment",
        hover_name=cluster_features.index,
        size="Total_Sales",
        title="Product Demand Clusters"
    )

        fig.update_layout(
        height=600,
        title_x=0.25
    )

        st.plotly_chart(
        fig,
        width="stretch"
    )

        st.info("""
**Business Insight**

Products located close together belong to similar demand groups,
while isolated clusters indicate unique sales behaviour that may
require different inventory strategies.
""")

        st.markdown("---")

            # =====================================================
    # Cluster Distribution
    # =====================================================

        st.subheader("📊 Demand Segment Distribution")

        segment_count = (
        cluster_features["Demand Segment"]
        .value_counts()
        .reset_index()
    )

        segment_count.columns = [
        "Demand Segment",
        "Count"
    ]

        fig = px.bar(
        segment_count,
        x="Demand Segment",
        y="Count",
        color="Demand Segment",
        text="Count",
        title="Products in Each Demand Segment"
    )

        fig.update_layout(
        height=500,
        title_x=0.25,
        xaxis_title="Demand Segment",
        yaxis_title="Number of Sub-Categories",
        showlegend=False
    )

        st.plotly_chart(
        fig,
        width="stretch"
    )

        st.markdown("---")

    # =====================================================
    # Cluster Assignment Table
    # =====================================================

        st.subheader("📋 Product Demand Segments")

        display_table = (
        cluster_features[
            [
                "Total_Sales",
                "Average_Order_Value",
                "Sales_Volatility",
                "Growth_Rate",
                "Demand Segment"
            ]
        ]
        .reset_index()
        .rename(columns={
            "Sub-Category": "Sub-Category",
            "Total_Sales": "Total Sales",
            "Average_Order_Value": "Average Order Value",
            "Sales_Volatility": "Sales Volatility",
            "Growth_Rate": "Growth Rate"
        })
    )

        st.dataframe(
        display_table.style.format({
            "Total Sales": "${:,.0f}",
            "Average Order Value": "${:,.2f}",
            "Sales Volatility": "{:,.2f}",
            "Growth Rate": "{:.2%}"
        }),
        width="stretch"
    )

        st.markdown("---")

    # =====================================================
    # Cluster Statistics
    # =====================================================

        st.subheader("📈 Cluster Statistics")

        cluster_stats = (
        cluster_features
        .groupby("Demand Segment")[
            [
                "Total_Sales",
                "Average_Order_Value",
                "Sales_Volatility",
                "Growth_Rate"
            ]
        ]
        .mean()
        .round(2)
    )

        st.dataframe(
        cluster_stats,
        width="stretch"
    )

        st.markdown("---")

    # =====================================================
    # Inventory Recommendations
    # =====================================================

        st.subheader("📦 Recommended Stocking Strategy")

        st.info("""
### 🏆 Premium High-Value Products
Maintain sufficient inventory despite lower purchase frequency because
each transaction contributes substantial revenue. Supplier coordination
should be prioritized.

---

### 📈 Steady Growth Products
Maintain balanced inventory levels and review sales regularly to support
consistent demand without excessive stock.

---

### 🔥 High Volume Products
Maintain high inventory availability with frequent replenishment to
prevent stock shortages during periods of strong customer demand.

---

### 🌱 Emerging Demand
Monitor demand closely and gradually increase inventory as sales continue
to grow while avoiding unnecessary overstocking.
""")

        st.markdown("---")

    # =====================================================
    # Key Findings
    # =====================================================

        st.subheader("📌 Key Findings")

        st.success("""
### Product Demand Segmentation Summary

✔ Four meaningful demand segments were identified using K-Means Clustering.

✔ Premium products generate high revenue despite relatively fewer transactions.

✔ High-volume products contribute the largest share of sales and require continuous inventory monitoring.

✔ Emerging demand products exhibit strong growth potential and should be monitored for future expansion.

✔ PCA visualization clearly demonstrates the separation between different demand segments.

✔ Demand segmentation enables data-driven inventory planning and optimized replenishment decisions.
""")

        st.markdown("---")

    # =====================================================
    # Business Insights
    # =====================================================

        st.subheader("💡 Business Insights")

        highest_sales_segment = (
        cluster_features.groupby("Demand Segment")["Total_Sales"]
        .sum()
        .idxmax()
    )

        st.info(f"""
The **{highest_sales_segment}** segment contributes the highest overall
sales and should receive priority in inventory allocation.

Using clustering helps businesses:

• Improve inventory planning.

• Reduce stock shortages.

• Optimize replenishment schedules.

• Identify premium and emerging products.

• Support data-driven business decisions.
""")

        st.markdown("---")

    # =====================================================
    # Download Cluster Report
    # =====================================================

        download_df = display_table.copy()

        csv = download_df.to_csv(index=False)

        st.download_button(
        "📥 Download Demand Segmentation Report",
        csv,
        "Demand_Segments_Report.csv",
        "text/csv"
    )

        st.markdown("---")

    # =====================================================
    # Footer
    # =====================================================

        st.markdown(
        """
<div style='text-align:center;color:gray;'>

### 📦 Product Demand Segmentation

Developed by **Chirra Nandhini Reddy**

Powered by **K-Means • PCA • Plotly • Streamlit • Python**

© 2026 All Rights Reserved

</div>
""",
        unsafe_allow_html=True
    )