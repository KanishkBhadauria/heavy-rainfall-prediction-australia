import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.features import add_features

# ---------- 1. Page setup ----------
st.set_page_config(page_title="Heavy Rain Predictor", page_icon="🌧️", layout="wide")

# ---------- 2. Load model (only once) ----------
@st.cache_resource
def load_model():
    model = joblib.load("models/final_model.joblib")
    with open("models/meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()
THRESHOLD = meta["threshold"]
MODERATE = 0.90
DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

# ---------- 3. Title + sidebar info ----------
st.title("🌧️ Heavy Rainfall Predictor: Australia")
st.write(f"Predicts whether **tomorrow's rainfall will exceed {meta['threshold_mm']} mm** "
         "at a weather station, using today's weather.")

with st.sidebar:
    st.header("ℹ️ Model info")
    st.write(f"**Model:** {meta['model']}")
    st.write(f"**Decision threshold:** {THRESHOLD:.3f}")
    st.write(f"**Test ROC-AUC:** {meta['test_roc_auc']}")
    st.write(f"**Test PR-AUC:** {meta['test_pr_auc']}")
    st.caption("Data: Australian Bureau of Meteorology, 2007–2017 (Kaggle weatherAUS)")

# ---------- 4. Input form ----------
st.subheader("Enter today's weather")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**📍 Place & rain**")
    location = st.selectbox("Location", meta["locations"],
                            index=meta["locations"].index("Cairns") if "Cairns" in meta["locations"] else 0)
    date = st.date_input("Today's date")
    rain_today = st.number_input("Rainfall today (mm)", 0.0, 400.0, 0.0, step=0.5)
    rain_yday  = st.number_input("Rainfall yesterday (mm)", 0.0, 400.0, 0.0, step=0.5)
    rain_2days = st.number_input("Rainfall 2 days ago (mm)", 0.0, 400.0, 0.0, step=0.5)

with c2:
    st.markdown("**🌡️ Temperature & humidity**")
    min_temp = st.number_input("Min temp (°C)", -10.0, 50.0, 18.0)
    max_temp = st.number_input("Max temp (°C)", -5.0, 50.0, 28.0)
    temp9  = st.number_input("Temp at 9am (°C)", -10.0, 50.0, 22.0)
    temp3  = st.number_input("Temp at 3pm (°C)", -10.0, 50.0, 27.0)
    hum9   = st.slider("Humidity 9am (%)", 0, 100, 70)
    hum3   = st.slider("Humidity 3pm (%)", 0, 100, 55)

with c3:
    st.markdown("**💨 Pressure, cloud & wind**")
    pres9  = st.number_input("Pressure 9am (hPa)", 970.0, 1045.0, 1013.0)
    pres3  = st.number_input("Pressure 3pm (hPa)", 970.0, 1045.0, 1010.0)
    cloud9 = st.slider("Cloud 9am (oktas, 0–8)", 0, 8, 4)
    cloud3 = st.slider("Cloud 3pm (oktas, 0–8)", 0, 8, 4)
    gust_dir = st.selectbox("Wind gust direction", DIRS, index=DIRS.index("SE"))
    gust_spd = st.number_input("Wind gust speed (km/h)", 0.0, 150.0, 40.0)
    dir9 = st.selectbox("Wind direction 9am", DIRS, index=DIRS.index("SE"))
    dir3 = st.selectbox("Wind direction 3pm", DIRS, index=DIRS.index("SE"))
    spd9 = st.number_input("Wind speed 9am (km/h)", 0.0, 100.0, 15.0)
    spd3 = st.number_input("Wind speed 3pm (km/h)", 0.0, 100.0, 20.0)
# ---------- 5. Predict ----------
if st.button("🔮 Predict tomorrow", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "Location": location, "Month": date.month,
        "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rain_today,
        "Evaporation": np.nan, "Sunshine": np.nan,
        "WindGustDir": gust_dir, "WindGustSpeed": gust_spd,
        "WindDir9am": dir9, "WindDir3pm": dir3,
        "WindSpeed9am": spd9, "WindSpeed3pm": spd3,
        "Humidity9am": hum9, "Humidity3pm": hum3,
        "Pressure9am": pres9, "Pressure3pm": pres3,
        "Cloud9am": cloud9, "Cloud3pm": cloud3,
        "Temp9am": temp9, "Temp3pm": temp3,
        "RainToday": "Yes" if rain_today > 1 else "No",
    }])

    row = add_features(row)
    row["Rain3dSum"] = rain_today + rain_yday + rain_2days

    score = float(model.predict_proba(row)[0, 1])
    heavy = score >= THRESHOLD

    if heavy:
        risk, color = "HIGH", "🔴"
    elif score >= MODERATE:
        risk, color = "MODERATE", "🟠"
    else:
        risk, color = "LOW", "🟢"

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Heavy rain tomorrow?", "YES ⚠️" if heavy else "NO ✅")
    r2.metric("Heavy rain probability", f"{score:.1%}",
              f"No heavy rain: {1 - score:.1%}", delta_color="off")
    r3.metric("Risk level", f"{color} {risk}")
    st.write(f"**Heavy rain probability: {score:.1%}** "
             f"(YES is shown when it reaches {THRESHOLD:.1%})")
    st.progress(min(score, 1.0))

    if heavy:
        st.error(f"⚠️ Heavy rainfall (> {meta['threshold_mm']} mm) is likely at {location} tomorrow.")
    elif risk == "MODERATE":
        st.warning("Conditions show some heavy-rain signals. Keep watching the forecast.")
    else:
        st.success("No heavy rainfall expected tomorrow.")

    st.caption("Note: the model score is a ranking score from a class-balanced model, "
               "not an exact chance of rain. On 2016-17 test data, about 1 in 4 'YES' "
               "warnings were correct. This is a student project, not an official forecast.")