import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATASET_PATH = "model_dataset.csv"
ARTIFACT_PATH = "model_artifacts.pkl"

#We split by season to avoid leaking future info. First few seasons to train, 2024 is to validate best model, 2025 is final test
TRAIN_MAX_SEASON = 2023
VAL_SEASON = 2024
TEST_SEASON = 2025

#Reads csv, has rolling diffs (home rolling stat - away rolling stat), travel, rest days, divisional, etc. and targets: margin and total_points
df = pd.read_csv(DATASET_PATH)

#Basic cleaning
df["season"] = pd.to_numeric(df["season"], errors="coerce")
df = df.dropna(subset=["season"]).copy()
df["season"] = df["season"].astype(int)

#Targets: margin = home_score - away_score (positive means home wins) and total_points = home_score + away_score (for over/under)
y_margin = df["margin"].astype(float)
y_total = df["total_points"].astype(float)

#Drop columns that we can't know pregame
DROP_COLS = {
    "margin",
    "total_points",
    "home_score",
    "away_score",
    "expected_result",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_starting_qb",
    "away_starting_qb",
}

features = [c for c in df.columns if c not in DROP_COLS]
X = df[features].copy()

#Split by season
train_mask = df["season"] <= TRAIN_MAX_SEASON
val_mask = df["season"] == VAL_SEASON
test_mask = df["season"] == TEST_SEASON

X_train, yM_train, yT_train = X[train_mask], y_margin[train_mask], y_total[train_mask]
X_val,   yM_val,   yT_val   = X[val_mask],   y_margin[val_mask],   y_total[val_mask]
X_test,  yM_test,  yT_test  = X[test_mask],  y_margin[test_mask],  y_total[test_mask]

#If empty then dataset doesn't contain that season
if len(X_train) == 0:
    raise RuntimeError("No training rows found. Check TRAIN_MAX_SEASON and your dataset seasons.")
if len(X_val) == 0:
    raise RuntimeError("No validation rows found. Check VAL_SEASON and your dataset seasons.")
if len(X_test) == 0:
    raise RuntimeError("No test rows found. Check TEST_SEASON and your dataset seasons.")

#MAE = Mean Absolute Error: average absolute difference between prediction and truth
#in our case: average points off (for margin or total)
#RMSE = Root Mean Squared Error which is like MAE but punishes big misses more
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

#Simple supervised model that learns from past games (we call them features) and tries to predict numeric target. 
#Ridge regression is basically linear regression with a penalty so it doesn't overfit as easily
def build_pipeline():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), #fill missing numbers
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)), #Model
    ])
    return pipe

#Trains pipeline
def fit_and_score(pipe, X_tr, y_tr, X_va, y_va, X_te, y_te):
    pipe.fit(X_tr, y_tr)

    pred_tr = pipe.predict(X_tr)
    pred_va = pipe.predict(X_va)
    pred_te = pipe.predict(X_te)

    tr_mae = mean_absolute_error(y_tr, pred_tr)
    va_mae = mean_absolute_error(y_va, pred_va)
    te_mae = mean_absolute_error(y_te, pred_te)

    tr_rmse = rmse(y_tr, pred_tr)
    va_rmse = rmse(y_va, pred_va)
    te_rmse = rmse(y_te, pred_te)

    return {
        "train_MAE": float(tr_mae),
        "val_MAE": float(va_mae),
        "test_MAE": float(te_mae),
        "train_RMSE": float(tr_rmse),
        "val_RMSE": float(va_rmse),
        "test_RMSE": float(te_rmse),
    }

#Predict margin
pipe_margin = build_pipeline()
metrics_margin = fit_and_score(pipe_margin, X_train, yM_train, X_val, yM_val, X_test, yM_test)

#Predict total points
pipe_total = build_pipeline()
metrics_total = fit_and_score(pipe_total, X_train, yT_train, X_val, yT_val, X_test, yT_test)

#sigma_margin is basically error scale for margin predictions
train_margin_pred = pipe_margin.predict(X_train)
sigma_margin = float(np.std(yM_train - train_margin_pred))
sigma_margin = max(sigma_margin, 10.0)

artifact = {
    #Matches feature names for upcoming rows
    "features": features,

    #margin and total model
    "pipeline_margin": pipe_margin,
    "pipeline_total": pipe_total,

    #Conversion
    "sigma_margin": sigma_margin,
}

joblib.dump(artifact, ARTIFACT_PATH)
print(f"Saved trained pipelines + metadata to: {ARTIFACT_PATH}")