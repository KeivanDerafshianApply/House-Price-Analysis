
"""
modeling.py
-----------
Readable modeling baselines for regression.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

@dataclass
class ModelReport:
    rmse_by_model: Dict[str, float]
    best_name: str

def get_models(random_state: int = 7) -> Dict[str, Any]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }

def evaluate(models: Dict[str, Any], X_valid, y_valid) -> Dict[str, float]:
    scores = {}
    for name, model in models.items():
        preds = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        scores[name] = float(rmse)
    return scores

def fit_and_select(X_train, y_train, X_valid, y_valid, random_state: int = 7) -> Tuple[Dict[str, Any], ModelReport]:
    models = get_models(random_state)
    for m in models.values():
        m.fit(X_train, y_train)
    scores = evaluate(models, X_valid, y_valid)
    best = min(scores, key=scores.get)
    return models, ModelReport(rmse_by_model=scores, best_name=best)
