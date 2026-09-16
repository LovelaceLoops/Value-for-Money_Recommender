import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import clean_data as data

X_train = data.X_train
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test
X = data.X
y = data.y

#Train model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print("\n" + "========== MODEL COEFFICIENTS ==========")
print("Intecept = " ,lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X_test.columns, columns=["Coefficient"])
print(coeff_df.sort_values(by="Coefficient", ascending=False))

#Evaluate Model
predictions = lm.predict(X_test)

sns.displot(y_test - predictions)
plt.title("Residual Distribution (Test Set)")
plt.xlabel("Actual − Predicted")
plt.show()

from sklearn import metrics
print("\n" + "========== MODEL PERFORMANCE (TEST SET) ==========")
print("MAE = ",metrics.mean_absolute_error(y_test,predictions))
print("MSE = ",metrics.mean_squared_error(y_test,predictions))
print("RMSE = ",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print("VAR = ",metrics.explained_variance_score(y_test,predictions))

#Value-for-Money ranking
predicted_prices = lm.predict(X)

sns.displot(y - predicted_prices)
plt.title("Residual Distribution (Full Dataset)")
plt.xlabel("Actual - Predicted")
plt.show()

results = pd.read_csv("cleaned_real_estate.csv")
results['Predicted Price'] = predicted_prices
results['Value Score'] = results['Predicted Price'] - results['Y house price of unit area']
results = results.sort_values(by='Value Score', ascending=False)

print("\n" + "========== TOP 10 BEST VALUE FOR MONEY PROPERTIES ==========")
display_cols = ['X2 house age', 'X3 distance to the nearest MRT station',
                 'X4 number of convenience stores', 'Y house price of unit area',
                 'Predicted Price', 'Value Score']
print(results[display_cols].head(10).to_string())