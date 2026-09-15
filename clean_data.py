import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
print("\n" + "========== LOAD REAL ESTATE DATASET ==========")
data = pd.read_csv("Real estate.csv")
print(data.head())
print(data.columns)
