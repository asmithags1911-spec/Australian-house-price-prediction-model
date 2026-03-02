# Australian-house-price-prediction-model
# 🏠 Surprise Housing — Australian Real Estate Price Prediction

> A production-ready machine learning pipeline that predicts house sale prices using the Ames Housing dataset. Built for Surprise Housing's expansion into the Australian real estate market.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [How to Run](#how-to-run)
6. [Pipeline Walkthrough](#pipeline-walkthrough)
7. [Models & Hyperparameter Tuning](#models--hyperparameter-tuning)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Key Findings](#key-findings)
10. [Output Files](#output-files)

---

## Project Overview

**Business Problem:** Surprise Housing wants to enter the Australian real estate market and needs a data-driven tool to:
- Predict the fair market value of prospective properties.
- Identify which features drive prices up or down.
- Minimize risk by avoiding overpaying or underpaying for properties.

**Solution:** A regularized regression and ensemble-based ML pipeline with thorough EDA, feature engineering, hyperparameter tuning, and multi-metric evaluation.

---

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | 1,460 rows × 81 columns (includes `SalePrice`) |
| `test.csv`  | 1,459 rows × 80 columns (no `SalePrice`) |
| `Data file.csv` | Optional extended dataset |
| `Datadescription.txt` | Column descriptions |

Download from: https://drive.google.com/drive/folders/1lXB1PUJThjo6DnnoH0QdvRf_qHMjJm2M

---

## Project Structure

```
surprise_housing/
│
├── house_price_prediction.py   # ← Main ML pipeline (all steps)
├── README.md                   # ← This file
│
├── train.csv                   # Put your data files here
├── test.csv
│
├── plots/                      # Auto-generated visualizations
│   ├── 01_target_distribution.png
│   ├── 02_feature_correlations.png
│   ├── 03_missing_values.png
│   ├── 04_scatter_top_features.png
│   ├── 05_model_comparison.png
│   ├── 06_*_feature_importance.png
│   ├── 07_*_residuals.png
│   └── 08_lasso_coefficients.png
│
├── models/                     # Saved model files (.pkl)
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── elasticnet_model.pkl
│   ├── randomforest_model.pkl
│   └── gradientboosting_model.pkl
│
├── submission.csv              # Final test predictions
└── model_results.csv           # Comparison table of all models
```

---

## Setup & Installation

**Python version:** 3.8+

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
scipy>=1.7
scikit-learn>=1.0
joblib>=1.1
```

---

## How to Run

1. Place `train.csv` and `test.csv` in the same folder as `house_price_prediction.py`.
2. Run:

```bash
python house_price_prediction.py
```

The script will:
- Load and validate data (falls back to synthetic demo data if files are missing).
- Perform EDA and save plots to `plots/`.
- Preprocess, feature-engineer, and encode the dataset.
- Train 5 models with GridSearchCV tuning.
- Print a full comparison table.
- Save models, results, and test predictions.

---

## Pipeline Walkthrough

### Step 1 — Data Loading
- Reads `train.csv` / `test.csv`.
- Prints shape, missing value counts, and column type breakdown.

### Step 2 — Exploratory Data Analysis (EDA)
- **Target distribution:** Raw vs log-transformed `SalePrice`, Q-Q plot.
- **Correlation heatmap:** Top 15 features most correlated with price.
- **Missing value chart:** Percentage of nulls per column.
- **Scatter plots:** Top correlated features vs `SalePrice`.

### Step 3 — Feature Engineering (New Features Created)

| Feature | Description |
|---------|-------------|
| `HouseAge` | Years since built at time of sale |
| `RemodAge` | Years since last remodel |
| `IsRemodeled` | Binary flag: was it remodeled? |
| `IsNew` | Binary: sold in year built |
| `TotalSF` | Basement + above-ground living area |
| `TotalBath` | Full + 0.5 × Half baths (basement + above) |
| `TotalPorchSF` | Sum of all porch/deck areas |
| `LivAreaPerRoom` | Living area efficiency per room |
| `QualCond` | OverallQual × OverallCond interaction |
| `HasPool` / `HasGarage` / `HasBsmt` / `HasFireplace` | Binary presence flags |
| `LogLotArea` / `LogGrLivArea` | Log-transformed area features |

### Step 4 — Data Cleaning
- Missing values for "absence" features (e.g., `PoolQC`, `GarageType`) filled with `'No'`.
- Remaining categorical NAs → mode imputation.
- Numerical NAs → median imputation.
- Outliers removed (e.g., very large homes with suspiciously low prices).

### Step 5 — Encoding
- **Ordinal features** (e.g., `ExterQual`, `KitchenQual`) → mapped to numeric scales (0–5).
- **Nominal features** → One-Hot Encoding (`pd.get_dummies`).
- Train/test column alignment after encoding to prevent feature mismatch.

### Step 6 — Skewness Correction
- Features with skewness > 0.75 are `log1p` transformed.
- Target `SalePrice` is also log-transformed.

### Step 7 — Train/Validation Split
- 80% train / 20% validation with `random_state=42`.

### Step 8 — Model Training & Tuning
See next section.

### Step 9 — Evaluation & Plots
- Residual plots, actual vs predicted, feature importance charts.
- Final predictions saved to `submission.csv`.

---

## Models & Hyperparameter Tuning

All models use **GridSearchCV** with 5-fold cross-validation and `neg_root_mean_squared_error` scoring.

### 1. Ridge Regression
- Regularization that shrinks coefficients but keeps all features.
- Tuned: `alpha ∈ [0.1, 1, 5, 10, 30, 50, 100, 200, 300]`

### 2. Lasso Regression
- Performs automatic feature selection by zeroing out irrelevant coefficients.
- Tuned: `alpha ∈ [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]`

### 3. ElasticNet
- Combines L1 (Lasso) + L2 (Ridge) penalties.
- Tuned: `alpha × l1_ratio` grid search.

### 4. Random Forest
- Ensemble of decision trees; robust to noise and non-linearity.
- Tuned: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`.

### 5. Gradient Boosting
- Sequential tree ensemble; typically best performer on tabular data.
- Tuned: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `min_samples_leaf`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE (log scale)** | Primary metric; lower is better. Penalizes large errors more. |
| **MAE (log scale)** | Mean absolute error; more interpretable average error. |
| **R² Score** | Variance explained; higher is better (max = 1.0). |
| **RMSE (original $)** | Real-world dollar prediction error. |
| **CV RMSE ± std** | Cross-validation score ± standard deviation. |

> **Why log-transform the target?**  
> `SalePrice` is right-skewed. Predicting `log(SalePrice)` makes errors more symmetric and prevents the model from being dominated by expensive homes.

---

## Key Findings

### Most Impactful Positive Drivers (Price ↑)
- `OverallQual` — Overall material and finish quality is the #1 predictor.
- `GrLivArea` / `TotalSF` — Larger living area = higher price.
- `GarageCars` / `GarageArea` — Garage capacity strongly matters.
- `YearBuilt` / `IsNew` — Newer homes command a premium.
- `Neighborhood` (StoneBr, NridgHt, NoRidge) — Location is decisive.
- `TotalBath` — More bathrooms, higher price.
- `ExterQual` / `KitchenQual` — Quality finishes add value.

### Most Impactful Negative Drivers (Price ↓)
- `HouseAge` / `RemodAge` — Older homes without remodeling lose value.
- `MSSubClass` (older/smaller styles) — Certain home types are discounted.
- `Functional` (deductions for damage) — Functional issues drop value sharply.
- `OverallCond` (poor condition) — Deferred maintenance hurts pricing.

### Lasso Feature Selection
Lasso typically zeros out 50–80% of engineered features, confirming that a small core set of ~30–50 features explains most price variation.

---

## Output Files

| File | Description |
|------|-------------|
| `submission.csv` | `Id, SalePrice` predictions for test set |
| `model_results.csv` | RMSE, MAE, R², CV scores for all 5 models |
| `models/*.pkl` | Saved model files (load with `joblib.load`) |
| `plots/*.png` | All EDA and model evaluation visualizations |

### Loading a saved model for inference:
```python
import joblib, numpy as np, pandas as pd

model = joblib.load('models/gradientboosting_model.pkl')
# X_new must go through the same preprocessing as training data
preds_log = model.predict(X_new)
preds_price = np.expm1(preds_log)
```

---

## Business Recommendations

1. **Focus acquisitions** on properties with high `OverallQual` (7+) in premium neighborhoods — these have the strongest price appreciation potential.
2. **Avoid** heavily aged homes (`HouseAge` > 50) without recent remodeling — they carry significant discount risk.
3. **Garage and living area** are the two easiest quantifiable value levers when comparing similar properties.
4. **Use the Lasso model** for explainability in stakeholder reports (transparent coefficients). Use **Gradient Boosting** for pure prediction accuracy.
5. Price predictions within ±10% of actual are achievable with the trained model, enabling confident bid/no-bid decisions.

---

*Project by: Data Science Team | Surprise Housing | 2024*
