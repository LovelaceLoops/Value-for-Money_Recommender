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
