"""
Surprise Housing - Australian Real Estate Market Price Prediction
=================================================================
A comprehensive machine learning pipeline for house price prediction
using Ridge, Lasso, ElasticNet, Random Forest, and Gradient Boosting.
"""

# ─────────────────────────────────────────────
# 1. IMPORTS & SETUP
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, norm

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Pipeline
from sklearn.pipeline import Pipeline

import joblib
import os

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 65)
print("  SURPRISE HOUSING - AUSTRALIAN REAL ESTATE PRICE PREDICTION")
print("=" * 65)


# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
def load_data(train_path='train.csv', test_path='test.csv'):
    """Load train and test datasets."""
    try:
        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)
        print(f"\n✔  Train shape : {train.shape}")
        print(f"✔  Test shape  : {test.shape}")
        return train, test
    except FileNotFoundError as e:
        print(f"\n⚠  File not found: {e}")
        print("   Generating synthetic demo data for illustration …")
        return generate_demo_data()


def generate_demo_data():
    """Generate synthetic Ames-like data for demonstration."""
    np.random.seed(42)
    n_train, n_test = 1460, 1459

    def make_df(n):
        df = pd.DataFrame({
            'Id'           : range(1, n + 1),
            'LotArea'      : np.random.randint(1500, 215245, n),
            'OverallQual'  : np.random.randint(1, 11, n),
            'OverallCond'  : np.random.randint(1, 10, n),
            'YearBuilt'    : np.random.randint(1872, 2011, n),
            'YearRemodAdd' : np.random.randint(1950, 2011, n),
            'TotalBsmtSF'  : np.random.randint(0, 6110, n),
            'GrLivArea'    : np.random.randint(334, 5642, n),
            'FullBath'     : np.random.randint(0, 4, n),
            'BedroomAbvGr' : np.random.randint(0, 8, n),
            'TotRmsAbvGrd' : np.random.randint(2, 15, n),
            'GarageCars'   : np.random.randint(0, 5, n),
            'GarageArea'   : np.random.randint(0, 1418, n),
            'Fireplaces'   : np.random.randint(0, 4, n),
            'MasVnrArea'   : np.random.randint(0, 1600, n).astype(float),
            'WoodDeckSF'   : np.random.randint(0, 857, n),
            'OpenPorchSF'  : np.random.randint(0, 547, n),
            'Neighborhood' : np.random.choice(['NAmes','CollgCr','OldTown','Edwards','Somerst'], n),
            'BldgType'     : np.random.choice(['1Fam','2fmCon','Duplex','TwnhsE','Twnhs'], n),
            'HouseStyle'   : np.random.choice(['2Story','1Story','1.5Fin','SLvl','SFoyer'], n),
            'SaleCondition': np.random.choice(['Normal','Abnorml','Partial','AdjLand','Alloca'], n),
            'ExterQual'    : np.random.choice(['Ex','Gd','TA','Fa'], n),
            'KitchenQual'  : np.random.choice(['Ex','Gd','TA','Fa'], n),
            'GarageType'   : np.random.choice(['Attchd','Detchd','BuiltIn','CarPort', np.nan], n),
            'MasVnrType'   : np.random.choice(['Stone','BrkFace','None', np.nan], n),
            'LotFrontage'  : np.where(np.random.rand(n) < 0.18, np.nan,
                                      np.random.randint(21, 313, n).astype(float)),
        })
        return df

    train = make_df(n_train)
    # Realistic price formula
    train['SalePrice'] = (
        train['GrLivArea']   * 55
        + train['OverallQual'] * 12000
        + train['GarageCars']  * 8000
        + train['TotalBsmtSF'] * 30
        + train['YearBuilt']   * 100
        - train['OverallCond'] * 1000
        + np.random.normal(0, 15000, n_train)
    ).clip(34900, 755000)

    test = make_df(n_test)
    return train, test


# ─────────────────────────────────────────────
# 3. EDA HELPER FUNCTIONS
# ─────────────────────────────────────────────
def eda_overview(df, name="Train"):
    print(f"\n{'─'*50}")
    print(f"  EDA OVERVIEW : {name}")
    print(f"{'─'*50}")
    print(f"Shape          : {df.shape}")
    print(f"Duplicate rows : {df.duplicated().sum()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print(f"\nTop missing columns ({len(missing)} total):")
        print(missing.head(15).to_string())
    print(f"\nNumerical columns  : {df.select_dtypes(include=np.number).shape[1]}")
    print(f"Categorical columns: {df.select_dtypes(include='object').shape[1]}")


def plot_target_distribution(y, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(y, bins=50, color='steelblue', edgecolor='white')
    axes[0].set_title('SalePrice Distribution (Raw)')
    axes[0].set_xlabel('SalePrice')

    y_log = np.log1p(y)
    axes[1].hist(y_log, bins=50, color='seagreen', edgecolor='white')
    axes[1].set_title('SalePrice Distribution (Log-transformed)')
    axes[1].set_xlabel('log(SalePrice + 1)')

    stats.probplot(y_log, plot=axes[2])
    axes[2].set_title('Q-Q Plot (Log SalePrice)')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✔  Saved: {save_dir}/01_target_distribution.png")


def plot_correlations(df, target='SalePrice', top_n=15, save_dir='plots'):
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()[target].drop(target).abs().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    corr.head(top_n).sort_values().plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(f'Top {top_n} Features Correlated with {target}')
    ax.set_xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔  Saved: {save_dir}/02_feature_correlations.png")
    return corr.head(top_n).index.tolist()


def plot_missing_values(df, save_dir='plots'):
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5, len(missing) * 0.3)))
    missing.plot(kind='barh', ax=ax, color='salmon', edgecolor='white')
    ax.set_title('Missing Value Percentage per Column')
    ax.set_xlabel('Missing %')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_missing_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔  Saved: {save_dir}/03_missing_values.png")


def plot_scatter_top_features(df, top_features, target='SalePrice', save_dir='plots'):
    features = [f for f in top_features[:6] if f in df.columns]
    n = len(features)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        axes[i].scatter(df[feat], df[target], alpha=0.4, color='steelblue', s=10)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'{feat} vs {target}')
    for j in range(i+1, 6):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/04_scatter_top_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔  Saved: {save_dir}/04_scatter_top_features.png")


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
COLS_NONE_MEANS_NO = [
    'PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
    'GarageType','GarageFinish','GarageQual','GarageCond',
    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'MasVnrType', 'MSSubClass'
]

ORDINAL_MAP = {
    'ExterQual'   : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0},
    'ExterCond'   : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0},
    'BsmtQual'    : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0,'No':0},
    'BsmtCond'    : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0,'No':0},
    'BsmtExposure': {'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0},
    'BsmtFinType1': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0,'No':0},
    'BsmtFinType2': {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0,'No':0},
    'HeatingQC'   : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0},
    'KitchenQual' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0},
    'FireplaceQu' : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0,'No':0},
    'GarageFinish': {'Fin':3,'RFn':2,'Unf':1,'NA':0,'No':0},
    'GarageQual'  : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0,'No':0},
    'GarageCond'  : {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0,'No':0},
    'PoolQC'      : {'Ex':4,'Gd':3,'TA':2,'Fa':1,'NA':0,'No':0},
    'Fence'       : {'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NA':0,'No':0},
    'LandSlope'   : {'Gtl':1,'Mod':2,'Sev':3},
    'LotShape'    : {'Reg':4,'IR1':3,'IR2':2,'IR3':1},
    'PavedDrive'  : {'Y':2,'P':1,'N':0},
    'Functional'  : {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0},
}


def engineer_features(df):
    """Create new features from existing ones."""
    df = df.copy()

    # Fill "None" for absent categorical features
    for col in COLS_NONE_MEANS_NO:
        if col in df.columns:
            df[col] = df[col].fillna('No')

    # Age-based features
    if 'YearBuilt' in df.columns:
        df['HouseAge']      = df.get('YrSold', 2010) - df['YearBuilt']
        df['RemodAge']      = df.get('YrSold', 2010) - df.get('YearRemodAdd', df['YearBuilt'])
        df['IsRemodeled']   = (df['YearBuilt'] != df.get('YearRemodAdd', df['YearBuilt'])).astype(int)
        df['IsNew']         = (df['YearBuilt'] == df.get('YrSold', 2010)).astype(int)

    # Area features
    if 'GrLivArea' in df.columns:
        df['TotalSF']       = df.get('TotalBsmtSF', 0) + df['GrLivArea']
        df['TotalPorchSF']  = (df.get('OpenPorchSF', 0) + df.get('EnclosedPorch', 0)
                               + df.get('3SsnPorch', 0)   + df.get('ScreenPorch', 0)
                               + df.get('WoodDeckSF', 0))
        df['LivAreaPerRoom']= df['GrLivArea'] / (df.get('TotRmsAbvGrd', 1) + 1)

    # Bath features
    df['TotalBath'] = (df.get('FullBath', 0)
                       + 0.5 * df.get('HalfBath', 0)
                       + df.get('BsmtFullBath', 0)
                       + 0.5 * df.get('BsmtHalfBath', 0))

    # Boolean flags
    df['HasPool']     = (df.get('PoolArea', pd.Series(0, index=df.index)) > 0).astype(int)
    df['HasGarage']   = (df.get('GarageArea', pd.Series(0, index=df.index)) > 0).astype(int)
    df['HasBsmt']     = (df.get('TotalBsmtSF', pd.Series(0, index=df.index)) > 0).astype(int)
    df['HasFireplace']= (df.get('Fireplaces', pd.Series(0, index=df.index)) > 0).astype(int)

    # Quality × Condition interaction
    if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
        df['QualCond'] = df['OverallQual'] * df['OverallCond']

    # Log of area (handle zeros)
    if 'LotArea' in df.columns:
        df['LogLotArea'] = np.log1p(df['LotArea'])
    if 'GrLivArea' in df.columns:
        df['LogGrLivArea'] = np.log1p(df['GrLivArea'])

    return df


def encode_ordinals(df):
    df = df.copy()
    for col, mapping in ORDINAL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def impute_numerical(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


def impute_categorical(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def encode_categorical(train_df, test_df):
    """One-hot encode remaining object columns, aligning train & test."""
    cat_cols = train_df.select_dtypes(include='object').columns.tolist()
    train_enc = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    test_enc  = pd.get_dummies(test_df,  columns=cat_cols, drop_first=True)
    train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)
    return train_enc, test_enc


def remove_outliers(df, target='SalePrice'):
    """Remove extreme outliers using IQR on key numeric features."""
    df = df.copy()
    # GrLivArea outliers (well-known in Ames dataset)
    if 'GrLivArea' in df.columns:
        df = df[~((df['GrLivArea'] > 4000) & (df[target] < 200000))]
    return df.reset_index(drop=True)


def fix_skewness(df, threshold=0.75):
    """Log1p transform skewed numeric features."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    skew_vals = df[num_cols].apply(lambda x: skew(x.dropna())).abs()
    skewed = skew_vals[skew_vals > threshold].index.tolist()
    # Don't transform target or negative-value columns
    safe = [c for c in skewed if df[c].min() >= 0]
    df[safe] = np.log1p(df[safe])
    print(f"\n✔  Log1p applied to {len(safe)} skewed features (threshold={threshold})")
    return df


# ─────────────────────────────────────────────
# 5. FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def preprocess(train_raw, test_raw):
    print("\n─── Preprocessing ───────────────────────────────────────")

    target_col = 'SalePrice'
    id_col     = 'Id'

    # Separate target
    y_raw  = train_raw[target_col].copy()
    y      = np.log1p(y_raw)        # log-transform target
    train  = train_raw.drop(columns=[target_col])
    test   = test_raw.copy()

    # Drop Id
    train_ids = train[id_col].values if id_col in train.columns else None
    test_ids  = test[id_col].values  if id_col in test.columns  else None
    train = train.drop(columns=[id_col], errors='ignore')
    test  = test.drop(columns=[id_col],  errors='ignore')

    # Combine for consistent processing
    n_train = len(train)
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)

    # Feature engineering
    combined = engineer_features(combined)
    combined = encode_ordinals(combined)
    combined = impute_numerical(combined)
    combined = impute_categorical(combined)

    # Split back
    train_proc = combined.iloc[:n_train].copy()
    test_proc  = combined.iloc[n_train:].copy()

    # Remove outliers from training only
    train_proc['SalePrice_tmp'] = y_raw.values
    train_proc = remove_outliers(train_proc, target='SalePrice_tmp')
    y = np.log1p(train_proc['SalePrice_tmp'])
    train_proc = train_proc.drop(columns=['SalePrice_tmp'])

    # Encode categoricals
    train_enc, test_enc = encode_categorical(train_proc, test_proc)

    # Fix skewness
    train_enc = fix_skewness(train_enc)
    test_enc  = fix_skewness(test_enc)

    # Align columns (post skewness fix)
    train_enc, test_enc = train_enc.align(test_enc, join='left', axis=1, fill_value=0)

    print(f"✔  Final train shape : {train_enc.shape}")
    print(f"✔  Final test shape  : {test_enc.shape}")
    print(f"✔  Target (log) stats: mean={y.mean():.3f}, std={y.std():.3f}")

    return train_enc, test_enc, y, train_ids, test_ids


# ─────────────────────────────────────────────
# 6. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, X_val, y_val, name='Model'):
    preds = model.predict(X_val)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    mae   = mean_absolute_error(y_val, preds)
    r2    = r2_score(y_val, preds)
    # On original scale
    rmse_orig = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(preds)))
    print(f"\n  [{name}]")
    print(f"    RMSE (log)    : {rmse:.5f}")
    print(f"    MAE  (log)    : {mae:.5f}")
    print(f"    R²            : {r2:.5f}")
    print(f"    RMSE (orig$)  : ${rmse_orig:,.0f}")
    return {'model': name, 'RMSE_log': rmse, 'MAE_log': mae, 'R2': r2, 'RMSE_orig': rmse_orig}


def cross_val_rmse(model, X, y, cv=5):
    scores = cross_val_score(model, X, y,
                              scoring='neg_root_mean_squared_error', cv=cv)
    return -scores.mean(), scores.std()


def build_models():
    """Return dict of model pipelines with hyperparameter grids."""
    models = {
        'Ridge': {
            'pipeline': Pipeline([
                ('scaler', RobustScaler()),
                ('model',  Ridge())
            ]),
            'param_grid': {
                'model__alpha': [0.1, 1, 5, 10, 30, 50, 100, 200, 300]
            }
        },
        'Lasso': {
            'pipeline': Pipeline([
                ('scaler', RobustScaler()),
                ('model',  Lasso(max_iter=10000))
            ]),
            'param_grid': {
                'model__alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
            }
        },
        'ElasticNet': {
            'pipeline': Pipeline([
                ('scaler', RobustScaler()),
                ('model',  ElasticNet(max_iter=10000))
            ]),
            'param_grid': {
                'model__alpha'   : [0.001, 0.005, 0.01, 0.05, 0.1],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        'RandomForest': {
            'pipeline': Pipeline([
                ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
            ]),
            'param_grid': {
                'model__n_estimators': [100, 200],
                'model__max_depth'   : [None, 15, 20],
                'model__min_samples_split': [2, 5],
                'model__max_features': ['sqrt', 0.5]
            }
        },
        'GradientBoosting': {
            'pipeline': Pipeline([
                ('model', GradientBoostingRegressor(random_state=42))
            ]),
            'param_grid': {
                'model__n_estimators'  : [200, 300],
                'model__learning_rate' : [0.05, 0.1],
                'model__max_depth'     : [3, 4],
                'model__subsample'     : [0.8, 1.0],
                'model__min_samples_leaf': [3, 5]
            }
        }
    }
    return models


def train_all_models(X_train, y_train, X_val, y_val, cv=5):
    print("\n" + "=" * 65)
    print("  MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 65)

    model_configs = build_models()
    results      = []
    best_models  = {}
    kf           = KFold(n_splits=cv, shuffle=True, random_state=42)

    for name, config in model_configs.items():
        print(f"\n  ▶  Training {name} …")
        gs = GridSearchCV(
            config['pipeline'],
            config['param_grid'],
            scoring='neg_root_mean_squared_error',
            cv=kf,
            n_jobs=-1,
            verbose=0
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        best_models[name] = best

        cv_rmse, cv_std = cross_val_rmse(best, X_train, y_train, cv=cv)
        print(f"     Best params  : {gs.best_params_}")
        print(f"     CV RMSE      : {cv_rmse:.5f} ± {cv_std:.5f}")

        metrics = evaluate_model(best, X_val, y_val, name=name)
        metrics['CV_RMSE']      = cv_rmse
        metrics['CV_RMSE_std']  = cv_std
        metrics['best_params']  = str(gs.best_params_)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values('RMSE_log')
    return best_models, results_df


def plot_model_comparison(results_df, save_dir='plots'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric in zip(axes, ['RMSE_log', 'R2', 'RMSE_orig']):
        results_df.sort_values(metric, ascending=(metric != 'R2')).plot(
            x='model', y=metric, kind='bar', ax=ax,
            color='steelblue', edgecolor='white', legend=False
        )
        ax.set_title(metric)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/05_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔  Saved: {save_dir}/05_model_comparison.png")


def plot_feature_importance(model, feature_names, model_name, top_n=20, save_dir='plots'):
    """Plot feature importances for tree models or coefficients for linear models."""
    final = model.named_steps['model']

    if hasattr(final, 'feature_importances_'):
        importances = pd.Series(final.feature_importances_, index=feature_names)
    elif hasattr(final, 'coef_'):
        importances = pd.Series(np.abs(final.coef_), index=feature_names)
    else:
        return

    top = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    top.plot(kind='barh', ax=ax, color='seagreen', edgecolor='white')
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}')
    ax.set_xlabel('Importance / |Coefficient|')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_{model_name.lower()}_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔  Saved: {save_dir}/06_{model_name.lower()}_feature_importance.png")
    return importances.nlargest(top_n)


def plot_residuals(model, X_val, y_val, model_name, save_dir='plots'):
    preds    = model.predict(X_val)
    residuals = y_val - preds

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(preds, residuals, alpha=0.4, color='steelblue', s=10)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted (log)')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name} — Residual Plot')

    axes[1].scatter(np.expm1(y_val), np.expm1(preds), alpha=0.4, color='seagreen', s=10)
    mn = min(np.expm1(y_val).min(), np.expm1(preds).min())
    mx = max(np.expm1(y_val).max(), np.expm1(preds).max())
    axes[1].plot([mn, mx], [mn, mx], 'r--')
    axes[1].set_xlabel('Actual Price ($)')
    axes[1].set_ylabel('Predicted Price ($)')
    axes[1].set_title(f'{model_name} — Actual vs Predicted')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/07_{model_name.lower()}_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔  Saved: {save_dir}/07_{model_name.lower()}_residuals.png")


# ─────────────────────────────────────────────
# 7. LASSO COEFFICIENT ANALYSIS
# ─────────────────────────────────────────────
def analyze_lasso_coefficients(lasso_model, feature_names, top_n=20, save_dir='plots'):
    coefs = pd.Series(
        lasso_model.named_steps['model'].coef_,
        index=feature_names
    )
    nonzero = coefs[coefs != 0].sort_values()

    pos = nonzero.nlargest(top_n)
    neg = nonzero.nsmallest(top_n)
    to_plot = pd.concat([neg, pos]).sort_values()

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['salmon' if v < 0 else 'steelblue' for v in to_plot]
    to_plot.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'Lasso Coefficients (Top {top_n} Positive & Negative)')
    ax.set_xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/08_lasso_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔  Saved: {save_dir}/08_lasso_coefficients.png")

    print(f"\n  Lasso: {(coefs == 0).sum()} features zeroed out of {len(coefs)}")
    print(f"\n  Top positive drivers (price↑):")
    print(pos.to_string())
    print(f"\n  Top negative drivers (price↓):")
    print(neg.to_string())
    return coefs


# ─────────────────────────────────────────────
# 8. GENERATE PREDICTIONS
# ─────────────────────────────────────────────
def generate_submission(best_model, X_test, test_ids, filename='submission.csv'):
    preds_log  = best_model.predict(X_test)
    preds_orig = np.expm1(preds_log)
    sub = pd.DataFrame({'Id': test_ids if test_ids is not None else range(1, len(preds_orig)+1),
                        'SalePrice': preds_orig})
    sub.to_csv(filename, index=False)
    print(f"\n✔  Submission saved to: {filename}")
    print(sub.describe())
    return sub


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # ── Load ──────────────────────────────────
    train_raw, test_raw = load_data('train.csv', 'test.csv')

    # ── EDA ───────────────────────────────────
    eda_overview(train_raw, "Train")
    eda_overview(test_raw,  "Test")

    plot_target_distribution(train_raw['SalePrice'])
    top_corr = plot_correlations(train_raw)
    plot_missing_values(train_raw)
    plot_scatter_top_features(train_raw, top_corr)

    # ── Preprocess ────────────────────────────
    X_all, X_test, y, train_ids, test_ids = preprocess(train_raw, test_raw)

    # ── Train / Validation split ──────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )
    print(f"\n✔  Train size : {X_train.shape[0]}  |  Val size : {X_val.shape[0]}")

    # ── Train models ──────────────────────────
    best_models, results_df = train_all_models(X_train, y_train, X_val, y_val)

    # ── Results summary ───────────────────────
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 65)
    print(results_df[['model','CV_RMSE','CV_RMSE_std','RMSE_log','R2','RMSE_orig']].to_string(index=False))

    plot_model_comparison(results_df)

    # ── Best model ────────────────────────────
    best_name  = results_df.iloc[0]['model']
    best_model = best_models[best_name]
    print(f"\n★  Best model : {best_name}")

    # ── Detailed analysis of best model ───────
    feat_names = X_all.columns.tolist()
    plot_feature_importance(best_model, feat_names, best_name)
    plot_residuals(best_model, X_val, y_val, best_name)

    # ── Lasso analysis (always) ───────────────
    analyze_lasso_coefficients(best_models['Lasso'], feat_names)

    # ── Also plot importance for all tree models ──
    for name in ['RandomForest', 'GradientBoosting']:
        if name in best_models:
            plot_feature_importance(best_models[name], feat_names, name)

    # ── Save models ───────────────────────────
    for name, model in best_models.items():
        joblib.dump(model, f'models/{name.lower()}_model.pkl')
    print("\n✔  All models saved to models/")

    # ── Submission ────────────────────────────
    generate_submission(best_model, X_test, test_ids, 'submission.csv')

    # ── Save results ──────────────────────────
    results_df.to_csv('model_results.csv', index=False)
    print("✔  model_results.csv saved")

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE ✓")
    print("=" * 65)


if __name__ == '__main__':
    main()
