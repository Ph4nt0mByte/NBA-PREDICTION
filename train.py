import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('games.csv')

# Target
df['HOME_WIN'] = df['HOME_TEAM_WINS']

# All available in-game stats (no PTS_diff to avoid leakage)
stats = ['FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

for stat in stats:
    df[f'{stat}_diff'] = df[f'{stat}_home'] - df[f'{stat}_away']

# Features
features = [f'{stat}_diff' for stat in stats]

X = df[features].dropna()
y = df['HOME_WIN'][X.index]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline for all models
scaler = StandardScaler()

# 1. Logistic Regression
lr_pipeline = Pipeline([('scaler', scaler), ('model', LogisticRegression(max_iter=2000))])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

# 2. Random Forest
rf_pipeline = Pipeline([('scaler', scaler), ('model', RandomForestClassifier(n_estimators=1000, random_state=42))])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

# 3. XGBoost (usually the strongest)
xgb_pipeline = Pipeline([('scaler', scaler), ('model', XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss'))])
xgb_pipeline.fit(X_train, y_train)
xgb_pred = xgb_pipeline.predict(X_test)

# Results
print("=== In-Game NBA Winner Analysis Results (All Stats + 3 Models) ===")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"Random Forest Accuracy:      {accuracy_score(y_test, rf_pred):.4f}")
print(f"XGBoost Accuracy:            {accuracy_score(y_test, xgb_pred):.4f}")

# Pick the best model
accuracies = {
    'Logistic': accuracy_score(y_test, lr_pred),
    'RandomForest': accuracy_score(y_test, rf_pred),
    'XGBoost': accuracy_score(y_test, xgb_pred)
}

best_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_name]

print(f"\nBest Model: {best_name} with {best_accuracy:.4f} accuracy")

best_pred = lr_pred if best_name == 'Logistic' else rf_pred if best_name == 'RandomForest' else xgb_pred
print("\nReport (Best Model):")
print(classification_report(y_test, best_pred))

# Save the best model
best_model = lr_pipeline if best_name == 'Logistic' else rf_pipeline if best_name == 'RandomForest' else xgb_pipeline
joblib.dump(best_model, 'nba_best_model.pkl')

# Auto-save metrics for UI
report_dict = classification_report(y_test, best_pred, output_dict=True)

metrics = {
    "logistic_accuracy": round(accuracy_score(y_test, lr_pred) * 100, 2),
    "random_forest_accuracy": round(accuracy_score(y_test, rf_pred) * 100, 2),
    "xgboost_accuracy": round(accuracy_score(y_test, xgb_pred) * 100, 2),
    "best_model": best_name,
    "overall_accuracy": round(best_accuracy * 100, 2),
    "classification_report": {
        "away_win": {
            "precision": round(report_dict['0']['precision'], 2),
            "recall": round(report_dict['0']['recall'], 2),
            "f1": round(report_dict['0']['f1-score'], 2),
            "support": int(report_dict['0']['support'])
        },
        "home_win": {
            "precision": round(report_dict['1']['precision'], 2),
            "recall": round(report_dict['1']['recall'], 2),
            "f1": round(report_dict['1']['f1-score'], 2),
            "support": int(report_dict['1']['support'])
        },
        "macro_avg": {
            "precision": round(report_dict['macro avg']['precision'], 2),
            "recall": round(report_dict['macro avg']['recall'], 2),
            "f1": round(report_dict['macro avg']['f1-score'], 2)
        },
        "weighted_avg": {
            "precision": round(report_dict['weighted avg']['precision'], 2),
            "recall": round(report_dict['weighted avg']['recall'], 2),
            "f1": round(report_dict['weighted avg']['f1-score'], 2)
        }
    }
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("\nBest model and metrics saved for UI!")
