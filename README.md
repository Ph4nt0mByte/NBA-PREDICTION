# 🏀 NBA In-Game Winner Analyzer

A machine learning application that predicts NBA game winners based on in-game statistics using multiple classification models (Logistic Regression, Random Forest, and XGBoost). The app provides real-time predictions with confidence scores through an interactive Streamlit interface.

## 📋 Overview

This project analyzes historical NBA game data to predict game outcomes based on in-game performance metrics. The system automatically selects the best-performing model and provides detailed performance metrics and classification reports.

**Current Best Model:** Logistic Regression with **83.69%** accuracy

## ✨ Features

- **Multi-Model Training**: Trains and compares three different machine learning models:
  - Logistic Regression
  - Random Forest (1000 estimators)
  - XGBoost (300 estimators)
- **Automatic Model Selection**: Selects the highest-performing model based on test accuracy
- **Interactive UI**: User-friendly Streamlit interface for real-time predictions
- **Detailed Performance Metrics**: Comprehensive classification reports with precision, recall, and F1-scores
- **Confidence Scoring**: Provides probability distributions for predictions
- **Data Leakage Prevention**: Uses only in-game stats (excludes point totals to avoid data leakage)

## 🎯 Prediction Features

The model uses the following in-game statistics (home minus away differentials):

- **FG_PCT_diff**: Field Goal Percentage Difference
- **FT_PCT_diff**: Free Throw Percentage Difference  
- **FG3_PCT_diff**: Three-Point Percentage Difference
- **AST_diff**: Assists Difference
- **REB_diff**: Rebounds Difference

## 📊 Dataset

The project includes the following CSV files:

- `games.csv` (4.1 MB): Game-level statistics and outcomes
- `games_details.csv` (93 MB): Detailed player-level game statistics
- `players.csv` (265 KB): Player information
- `ranking.csv` (15.4 MB): Team ranking history
- `teams.csv` (4 KB): Team information

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd NBA
```

2. Install required dependencies:
```bash
pip install streamlit pandas scikit-learn xgboost joblib
```

### Usage

#### 1. Train the Models

First, train the models and generate performance metrics:

```bash
python train.py
```

This will:
- Load and preprocess the NBA game data
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate and compare model performance
- Save the best model as `nba_best_model.pkl`
- Generate `model_metrics.json` with detailed performance metrics

#### 2. Run the Web Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

#### 3. Make Predictions

1. Use the sliders to input in-game stat differences (home team minus away team):
   - FG % Difference
   - FT % Difference
   - 3PT % Difference
   - Assists Difference
   - Rebounds Difference

2. Enter team names (e.g., "Lakers" for home team, "Celtics" for away team)

3. Click "Analyze Winner" to get:
   - Winner prediction
   - Confidence percentage
   - Probability distribution chart

## 📈 Model Performance

Current performance metrics (auto-updated from latest training):

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 83.69% |
| Logistic Regression | 83.69% |
| Random Forest | 82.88% |
| XGBoost | 81.96% |

### Classification Report (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Away Win | 0.81 | 0.79 | 0.80 | 2,182 |
| Home Win | 0.86 | 0.87 | 0.86 | 3,129 |
| **Macro Avg** | 0.83 | 0.83 | 0.83 | - |
| **Weighted Avg** | 0.84 | 0.84 | 0.84 | - |

## 🗂️ Project Structure

```
NBA/
├── app.py                    # Streamlit web application
├── train.py                  # Model training script
├── games.csv                 # Game statistics dataset
├── games_details.csv         # Detailed player statistics
├── players.csv               # Player information
├── ranking.csv               # Team rankings
├── teams.csv                 # Team information
├── nba_best_model.pkl        # Best trained model (auto-generated)
├── nba_future_model.pkl      # Alternative model variant
├── nba_ingame_model.pkl      # Alternative model variant
├── nba_max_model.pkl         # Alternative model variant (8.5 MB)
├── model_metrics.json        # Performance metrics (auto-generated)
└── README.md                 # This file
```

## 🔬 Methodology

### Data Preprocessing
- Calculates differential statistics (home - away) for each feature
- Removes records with missing values
- Stratified train-test split (80/20)

### Feature Engineering
- Uses only in-game performance metrics
- Excludes point totals to prevent data leakage
- StandardScaler normalization for all models

### Model Training
- All models use sklearn pipelines with StandardScaler
- Cross-validation with stratified sampling
- Random state = 42 for reproducibility

### Evaluation
- Accuracy score for model selection
- Detailed classification reports
- Probability calibration for confidence scores

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn, XGBoost
- **Data Processing**: pandas
- **Model Persistence**: joblib
- **Metrics Storage**: JSON

## 📝 Notes

- The model uses **in-game statistics only** to avoid data leakage from final scores
- Point differentials are intentionally excluded from features
- Models are trained on historical NBA data with proper train-test splitting
- The best model is automatically selected and deployed
- Metrics are automatically saved and displayed in the UI

## 🔄 Retraining

To retrain the models with updated data or different parameters:

1. Update the CSV files with new data
2. Modify hyperparameters in `train.py` if desired
3. Run `python train.py`
4. Restart the Streamlit app to load the new model

## 📄 License

This project is provided as-is for educational and analytical purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or bug fixes.

---

**Note**: This is a predictive analytics tool for educational purposes. Sports outcomes involve many unpredictable factors, and predictions should not be used for gambling or betting purposes.
