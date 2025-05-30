# Retail Sales Forecasting Project

This project demonstrates a comprehensive approach to forecasting retail sales using a variety of time series and machine learning models. The workflow covers data preprocessing, exploratory analysis, classical statistical models, advanced machine learning models, and model evaluation.

## Project Structure

- **Data Preparation:**  
    - Load and preprocess sales and store data.
    - Feature engineering (date features, promotions, holidays, etc.).

- **Classical Time Series Models:**  
    - Naive, Moving Average, Seasonal Naive, Weighted Moving Average.

- **Exponential Smoothing (ETS) Models:**  
    - Simple Exponential Smoothing, Holt’s Linear Trend, Holt-Winters Seasonal.

- **ARIMA & SARIMAX:**  
    - Stationarity checks (ADF, KPSS), autocorrelation analysis, ARIMA and SARIMAX modeling.

- **Prophet Model:**  
    - Facebook Prophet for trend and seasonality modeling.

- **Machine Learning Models:**  
    - Random Forest, XGBoost, GradientBoosting, LightGBM.
    - Feature selection and model training.

- **Model Evaluation:**  
    - Metrics: RMSE, MAE, MAPE, MdRAE, GMRAE, RMSPE.
    - Visual comparison of predictions vs. actuals.
    - Summary tables for model performance.

- **Final Prediction & Submission:**  
    - Use the best-performing model to predict sales on the test set.
    - Prepare submission file for Kaggle.

## How to Run

1. **Install Requirements:**  
     - Python 3.x  
     - pandas, numpy, matplotlib, scikit-learn, statsmodels, prophet, xgboost, lightgbm

2. **Prepare Data:**  
     - Place `train.csv`, `test.csv`, and `store.csv` in the working directory.

3. **Run the Notebook:**  
     - Execute each cell in order for full workflow and results.

4. **Submission:**  
     - The notebook generates a `submission.csv` file for Kaggle submission.

## Key Results

- Machine learning models (especially GradientBoosting and LightGBM) achieved the best accuracy.
- Classical and statistical models serve as useful baselines and for interpretability.
- The workflow includes robust evaluation and comparison of all models.

## Acknowledgements

- Data from Kaggle Rossmann Store Sales competition.
- Libraries: pandas, numpy, matplotlib, scikit-learn, statsmodels, prophet, xgboost, lightgbm.

---
