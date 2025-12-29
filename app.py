import streamlit as st
import joblib
import pandas as pd
import json
import os

# Load best model
model = joblib.load('nba_best_model.pkl')

# Load metrics
if os.path.exists('model_metrics.json'):
    with open('model_metrics.json', 'r') as f:
        m = json.load(f)
else:
    st.error("Run train.py first!")
    st.stop()

st.title("🏀 NBA In-Game Winner Analyzer (Maximum Accuracy)")

# Sidebar
st.sidebar.header("📊 Exact Model Performance")
st.sidebar.write(f"**Best Model:** {m['best_model']}")
st.sidebar.metric("Overall Accuracy", f"{m['overall_accuracy']}%")
st.sidebar.metric("Logistic Regression", f"{m['logistic_accuracy']}%")
st.sidebar.metric("Random Forest", f"{m['random_forest_accuracy']}%")
st.sidebar.metric("XGBoost", f"{m['xgboost_accuracy']}%")

with st.sidebar.expander("Full Classification Report"):
    r = m['classification_report']
    data = {
        "Class": ["Away Win", "Home Win", "Macro Avg", "Weighted Avg"],
        "Precision": [f"{r['away_win']['precision']:.2f}", f"{r['home_win']['precision']:.2f}", f"{r['macro_avg']['precision']:.2f}", f"{r['weighted_avg']['precision']:.2f}"],
        "Recall": [f"{r['away_win']['recall']:.2f}", f"{r['home_win']['recall']:.2f}", f"{r['macro_avg']['recall']:.2f}", f"{r['weighted_avg']['recall']:.2f}"],
        "F1-Score": [f"{r['away_win']['f1']:.2f}", f"{r['home_win']['f1']:.2f}", f"{r['macro_avg']['f1']:.2f}", f"{r['weighted_avg']['f1']:.2f}"],
        "Support": [str(r['away_win']['support']), str(r['home_win']['support']), "-", "-"]
    }
    df = pd.DataFrame(data)
    st.table(df)

# Input section
st.write("Enter **in-game stat differences** (home minus away) for maximum accuracy.")

col1, col2 = st.columns(2)
with col1:
    fg_pct_diff = st.slider("FG % Diff", -0.3, 0.3, 0.0, step=0.01)
    ft_pct_diff = st.slider("FT % Diff", -0.3, 0.3, 0.0, step=0.01)
    fg3_pct_diff = st.slider("3PT % Diff", -0.3, 0.3, 0.0, step=0.01)

with col2:
    ast_diff = st.slider("Assists Diff", -30, 30, 0)
    reb_diff = st.slider("Rebounds Diff", -30, 30, 0)

home_team = st.text_input("Home Team (e.g., Lakers)")
away_team = st.text_input("Away Team (e.g., Celtics)")

if st.button("Analyze Winner"):
    input_data = pd.DataFrame({
        'FG_PCT_diff': [fg_pct_diff],
        'FT_PCT_diff': [ft_pct_diff],
        'FG3_PCT_diff': [fg3_pct_diff],
        'AST_diff': [ast_diff],
        'REB_diff': [reb_diff]
    })

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    winner = home_team if pred == 1 else away_team
    confidence = prob[1] if pred == 1 else prob[0]

    st.success(f"**{winner} wins!** with {confidence:.0%} confidence")
    st.bar_chart({"Home Win Prob": prob[1], "Away Win Prob": prob[0]})

st.caption("Maximum accuracy using all in-game stats (no points leakage). Metrics auto-update from train.py.")
