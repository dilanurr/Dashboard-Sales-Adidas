# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import streamlit as st
import plotly.express as px

plt.style.use("seaborn-v0_8")

# %% [markdown]
# Load Data

# %%
file_path = "Adidas US Sales Datasets.xlsx"
df = pd.read_excel(file_path, sheet_name="Data Sales Adidas", header=4)

# %% [markdown]
# Cleaning Data

# %%
df.info()

# %%
# Delete Column unnamed
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", na=False)]

# %%
# Standarisasi Column Name
df.columns = (
    df.columns.str.strip()
             .str.lower()
             .str.replace(r"[^\w\s]", "", regex=True)
             .str.replace(r"\s+", "_", regex=True)
)

# %%
# Calculate na per column
na_column = df.isna().sum()
print(na_column)

# %%
# Pastikan kolom tanggal sesuai format
df["invoice_date"] = pd.to_datetime(df["invoice_date"])
df["year"] = df["invoice_date"].dt.year
df["month"] = df["invoice_date"].dt.month
df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)
df["date_str"] = df["invoice_date"].dt.strftime("%Y-%m-%d")

# %%
df.info()


# %%

# %% [markdown]
# Update Dashboard Function

# %%
def update_dashboard(change=None):
    clear_output(wait=True)
    display(region_widget, product_widget, year_widget, method_widget)


# %%
# Filter data
st.sidebar.header("Filter Data")

region_filter = st.sidebar.multiselect(
    "Choose Region", options=df["region"].unique(), default=df["region"].unique()
)
product_filter = st.sidebar.multiselect(
    "Choose Product", options=df["product"].unique(), default=df["product"].unique()
)
year_filter = st.sidebar.multiselect(
    "Choose Tahun", options=df["year"].unique(), default=df["year"].unique()
)
sales_method_filter = st.sidebar.multiselect(
    "Choose Sales Method", options=df["sales_method"].unique(), default=df["sales_method"].unique()
)

# %%
# Filter Data Frame
filtered_df = df[
    (df["region"].isin(region_filter)) &
    (df["product"].isin(product_filter)) &
    (df["year"].isin(year_filter)) &
    (df["sales_method"].isin(sales_method_filter))
]

# %%

# %% [markdown]
# KPI

# %%
total_sales = filtered_df["total_sales"].sum()
total_units = filtered_df["units_sold"].sum()
total_profit = filtered_df["operating_profit"].sum()
profit_margin = (total_profit / total_sales) * 100 if total_sales != 0 else 0

# %%
st.title("Adidas US Sales Dashboard")

# %%
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Units Sold", f"{total_units:,.0f}")
col3.metric("Operating Profit", f"${total_profit:,.0f}")
col4.metric("Profit Margin", f"{profit_margin:.2f}%")

# %%

# %% [markdown]
# Chart

# %%
# Monthly Sales Trends
sales_trend = filtered_df.groupby("year_month")["total_sales"].sum().reset_index()
fig_trend = px.line(sales_trend, x="year_month", y="total_sales", title="Monthly Sales Trends")
st.plotly_chart(fig_trend, use_container_width=True)

# %%
# Sales Per Product
sales_product = filtered_df.groupby("product")["total_sales"].sum().reset_index()
fig_product = px.bar(sales_product, x="product", y="total_sales", title="Sales of Product")
st.plotly_chart(fig_product, use_container_width=True)

# %%
# Sales of Region
sales_region = filtered_df.groupby("region")["total_sales"].sum().reset_index()
fig_region = px.pie(sales_region, names="region", values="total_sales", title="Distribution Sales of Region")
st.plotly_chart(fig_region, use_container_width=True)


# %%
# Scatterplot Corelation Price Vc Unit Sold
fig_scatter = px.scatter(
    filtered_df,
    x="price_per_unit",
    y="units_sold",
    color="product",
    size="total_sales",
    hover_data=["region", "sales_method"],
    title="Corelation Price Vc Unit Sold",
    size_max=15  
)

# Add dollar symbol on X axis
fig_scatter.update_layout(
    xaxis=dict(tickprefix="$"),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="white")
)

st.plotly_chart(fig_scatter, use_container_width=True)

# %%
# Top 5 Product
top_products = (
    filtered_df.groupby("product")["total_sales"]
    .sum()
    .reset_index()
    .sort_values(by="total_sales", ascending=False)
    .head(5)  
)

# bar chart
fig_top_products = px.bar(
    top_products,
    x="product",
    y="total_sales",
    title="Top 5 Best Selling Products",
    text_auto=".2s",  
    color="total_sales",
    color_continuous_scale="Blues"  
)

st.plotly_chart(fig_top_products, use_container_width=True)

# %%

# %% [markdown]
# Tampilkan Data

# %%
st.subheader("Latest Data (After Filter))")
st.dataframe(filtered_df)

# %%

# %%

# %%

# %%
# !pip install jupytext


# %%
# import jupytext
import sys
# !{sys.executable} -m pip install jupytext

# %%
# jupytext.configure_notebook_metadata(fmt="ipynb,py")
# !{sys.executable} -m jupytext --set-formats ipynb,py DashboardAdidasSales.ipynb

# %%
#Ceks path
import os
print(os.path.abspath("DashboardAdidasSales.py"))

# %% [raw]
# cd "C:\Users\dilan\My Project"
# streamlit run DashboardAdidasSales.py 
# #ketik di anaconda prompt

# %%

# %%
