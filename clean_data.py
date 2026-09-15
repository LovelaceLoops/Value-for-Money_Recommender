import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
print("\n" + "========== LOAD REAL ESTATE DATASET ==========")
data = pd.read_csv("Real estate.csv")
print(data.head())
print(data.columns)

#Check for outliers and understand spread of price data
print("\n" + "========== OUTLIERS DETECTION AND PRICE DATA ==========")
plt.figure(figsize=(6, 4))
sns.boxplot(y=data['Y house price of unit area'], color='#4C72B0')
plt.title("Price Distribution — Outlier Check")
plt.tight_layout()
plt.show()

#Heatmap to verify correlation of price data and other components
print("\n" + "========== CORRELATION MATRIX ==========")
print(data[['X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
       'Y house price of unit area']].corr())

sns.set_style("white")

plt.figure(figsize=(9, 7), dpi=100)
sns.heatmap(
    data[['X1 transaction date', 'X2 house age',
          'X3 distance to the nearest MRT station',
          'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
          'Y house price of unit area']].corr(),
    annot=True,
    fmt=".2f",
    cmap='vlag',
    center=0,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Correlation'},
    annot_kws={"size": 9}
)
plt.title("Correlation with Price", fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)

#Data Visualization
print("\n" + "========== HOUSE PRICES PER UNIT AREA ==========")
sns.set_style("whitegrid")  
plt.figure(figsize=(8, 5))
plt.hist(data["Y house price of unit area"], bins=30, color="#4C72B0", edgecolor='white', alpha=0.85)
plt.title("Distribution of House Prices (per Unit Area)", fontsize=13, fontweight='bold')
plt.xlabel("Price per Unit Area", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.show()

#Check for null values
print("\n" + "========== NULL VALUES DETECTION ==========")
plt.figure(figsize=(8, 5), dpi=100)
sns.heatmap(
    data.isnull(),
    yticklabels=False,
    cbar=False,
    cmap="viridis"
)
plt.title("Missing Value Check", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Columns", fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.show()
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()

#Drop useless columns
print("\n" + "========== DROP 'No' COLUMN ==========")
data = data.drop('No',axis=1)
