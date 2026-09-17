# Value-for-Money_Recommender

## Author

Created by **Mrunal Patil & Kartik Patil** as part of an engineering coursework project.

A content-based recommender that predicts a property's expected price using Linear Regression, then flags properties priced **below** their predicted value as good deals. Properties are ranked by a **Value Score** — the gap between what the model expects a property to cost and what it's actually listed at.

## Dataset

**Source:** [Real Estate Price Prediction](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction) (Kaggle), based on real Taiwan real-estate transaction data.

- **414 rows**, 6 usable features, no missing values
- **Target:** `Y house price of unit area`
- **Features:**
  - `X1 transaction date`
  - `X2 house age`
  - `X3 distance to the nearest MRT station`
  - `X4 number of convenience stores`
  - `X5 latitude`
  - `X6 longitude`

An earlier version of this project used a synthetic car-pricing dataset. It was dropped after correlation analysis showed price had essentially no relationship with mileage or model year (r ≈ 0.01) — the dataset carried a "for learning purposes only" disclaimer and wasn't designed to reflect real-world pricing. This real-estate dataset was chosen instead because it shows genuine, learnable relationships between price and its features.

## Approach

1. **Data cleaning** — load the CSV, drop the row-identifier column (`No`), check for missing values
2. **Exploratory analysis** — boxplot (outlier check), correlation heatmap, price distribution histogram
3. **Feature/target split** — `X` = the 6 features above, `y` = price
4. **Train/test split** — 80/20, `random_state=101`
5. **Model training** — scikit-learn `LinearRegression`
6. **Evaluation** — MAE, MSE, RMSE, and explained variance (VAR) on the held-out test set
7. **Value-for-money scoring** — predict price for every property, then rank by `Predicted Price − Actual Price`

## Correlation with Price

| Feature | Correlation |
|---|---|
| Distance to nearest MRT station | -0.67 |
| Number of convenience stores | 0.57 |
| Latitude | 0.55 |
| Longitude | 0.52 |
| House age | -0.21 |
| Transaction date | 0.09 |

Distance to the nearest MRT station is the strongest single predictor of price.

## Model Performance (test set)

| Metric | Value |
|---|---|
| MAE | 5.20 |
| MSE | 44.13 |
| RMSE | 6.64 |
| VAR (explained variance) | 0.625 (62.5%) |

The model explains roughly 62.5% of the variation in price using just 6 features — a solid result given the dataset doesn't include factors like interior condition, floor level, or renovation status.

## How the Recommendation Works

For every property, the model predicts an "expected" price based on its features. The **Value Score** is:

```
Value Score = Predicted Price − Actual Price
```

- **Positive score** → priced below what the model expects → a good deal
- **Negative score** → priced above what the model expects → possibly overpriced

Properties are sorted by Value Score, descending, so the top of the list is the best value-for-money recommendation.

## File Structure

```
clean_data.py    # loads, cleans, and visualizes the raw dataset;
                  # saves a cleaned CSV for modeling
train_model.py    # loads the cleaned CSV, trains and evaluates the model,
                  # and generates the value-for-money ranking
Real estate.csv   # raw dataset
```

**Run order:**
```
python clean_data.py
python train_model.py
```

## Requirements

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Limitations

- Only 6 features are available — no interior condition, floor level, view, or renovation status, which would likely improve accuracy
- `X3`–`X6` (distance to MRT, convenience stores, latitude, longitude) are moderately correlated with each other, since they all partly capture "how central the property is" — this can make individual coefficients harder to interpret in isolation, though it doesn't hurt overall prediction accuracy
- The top-ranked property in the value-for-money list was checked against the full price distribution and confirmed to be a genuine low-price property (not a data entry error), though it remains a notable outlier worth flagging when interpreting results
