"""
Train and evaluate the House Price Prediction model.
The built-in California Housing dataset downloads the first time you run this script.
Internet is needed for the initial download.
"""
import json
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "house_price_model.pkl"
METRICS_PATH = "model_metrics.json"
RANDOM_STATE = 42

def main():
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target  # units of $100,000

    print(f"Rows: {len(X):,}")
    print("Features:", ", ".join(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=120,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions) * 100000
    rmse = mean_squared_error(y_test, predictions) ** 0.5 * 100000
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    metrics = {
        "dataset": "California Housing (scikit-learn)",
        "rows": int(len(X)),
        "features": list(X.columns),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "mae_usd": round(float(mae), 2),
        "rmse_usd": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "target_unit": "USD, converted from the dataset's $100,000 units"
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete!")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"MAE: ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R²: {r2:.3f}")
    print("\nNote: These are evaluation metrics for California district median values, not Indian individual house prices.")

if __name__ == "__main__":
    main()
