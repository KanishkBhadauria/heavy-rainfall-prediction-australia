# 🌧️ Heavy Rainfall Prediction: Australia

Predicts whether **tomorrow's rainfall will exceed 64.5 mm** (IMD "heavy rain" threshold) at an Australian weather station, using today's weather observations.

## 📊 Dataset
- **Rain in Australia (weatherAUS)**, Kaggle: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
- Source: Australian Bureau of Meteorology
- 2007–2017 · 49 stations · 145,460 rows · 23 columns
- After cleaning: 142,017 rows · **463 heavy-rain days (0.33%)**, so the data is highly imbalanced

## 💻 Tech stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit

## 🔄 Method
1. **Target:** HeavyRainTomorrow = 1 if the next day's rainfall at the same station is > 64.5 mm
2. **Leakage prevention:** removed `RainTomorrow`; time-based split
3. **Split:** Train 2007–2014 · Validation 2015 · Test 2016–2017
4. **Preprocessing:** median/mode imputation, one-hot encoding, standard scaling
5. **New features:** TempRange, HumidityChange, PressureChange, Rain3dSum, Month_sin, Month_cos
6. **Models:** Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost (with class weighting)
7. **Model selection:** by PR-AUC (accuracy is misleading on imbalanced data)
8. **Threshold tuning:** on the validation set, maximising F1

## 📈 Results

### Validation (2015), threshold 0.5
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.913 | 0.034 | 0.914 | 0.066 | **0.967** | **0.274** |
| Random Forest | 0.996 | 0.360 | 0.155 | 0.217 | 0.962 | 0.212 |
| KNN | 0.997 | 0.000 | 0.000 | 0.000 | 0.824 | 0.176 |
| XGBoost | 0.990 | 0.150 | 0.448 | 0.225 | 0.945 | 0.157 |
| Decision Tree | 0.932 | 0.038 | 0.793 | 0.073 | 0.877 | 0.080 |

**Best model:** Logistic Regression.
**Tuned threshold:** 0.995, which raised validation F1 from 0.066 to **0.400** (Precision 0.373, Recall 0.431).

### Final test (2016–2017)
| Metric | Value |
|---|---|
| ROC-AUC | **0.936** |
| PR-AUC | **0.179** (random baseline 0.0033) |
| Heavy-rain Precision | 0.23 |
| Heavy-rain Recall | 0.21 |

## ⚠️ Limitations
- Heavy rain is very rare (0.33%), so results change a lot with just a few storms
- The very high threshold (0.995) is sensitive to small shifts in model probabilities
- Test scores are lower than validation because weather differs between years
- Future work: probability calibration, XGBoost tuning, more recent data

## ▶️ How to run
1. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Train the 5 models (saves to `models/` and `reports/`):
   ```
   python -m src.train
   ```
3. Open `04_final_model.ipynb` and click **Run All** (creates `models/final_model.joblib` and `models/meta.json`)
4. Start the web app:
   ```
   streamlit run app.py
   ```
5. Open http://localhost:8501 in your browser

Note: `KNN.joblib` (113 MB) is not included because it exceeds GitHub's file size limit. Run `python -m src.train` to recreate it.

## 📁 Project structure
```
data/        weatherAUS.csv
src/         preprocessing.py (cleaning + target) · features.py (feature engineering) · train.py (model training)
models/      trained models + final_model.joblib + meta.json
reports/     charts + model_comparison.csv
step1_explore.ipynb · 02_eda.ipynb · 03_models.ipynb · 04_final_model.ipynb
app.py       Streamlit web app
requirements.txt
```