import numpy as np
from src.preprocessing import BASE_NUM

NEW_FEATURES = ["TempRange", "HumidityChange", "PressureChange",
                "Rain3dSum", "Month_sin", "Month_cos"]

# All numeric columns the model should use (original + new)
FEATURE_COLS = BASE_NUM + ["Month"] + NEW_FEATURES

def add_features(df):
    df = df.copy()
    df["TempRange"]      = df["MaxTemp"] - df["MinTemp"]
    df["HumidityChange"] = df["Humidity3pm"] - df["Humidity9am"]
    df["PressureChange"] = df["Pressure3pm"] - df["Pressure9am"]
    df["Rain3dSum"] = (df.groupby("Location")["Rainfall"]
                         .transform(lambda s: s.rolling(3, min_periods=1).sum()))
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df