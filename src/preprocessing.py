import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

THRESHOLD = 64.5
TARGET = "HeavyRainTomorrow"
BASE_NUM = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
            "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
            "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
            "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]
CAT_COLS = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

def load_and_clean(path="data/weatherAUS.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)
    g = df.groupby("Location")
    df["RainfallTomorrow"] = g["Rainfall"].shift(-1)
    gap = (g["Date"].shift(-1) - df["Date"]).dt.days
    df.loc[gap != 1, "RainfallTomorrow"] = None
    df = df.dropna(subset=["RainfallTomorrow"])
    df[TARGET] = (df["RainfallTomorrow"] > THRESHOLD).astype(int)
    df["Month"] = df["Date"].dt.month
    return df.drop(columns=["RainTomorrow", "RainfallTomorrow"])

def time_split(df):
    train = df[df["Date"] < "2015-01-01"]
    val   = df[(df["Date"] >= "2015-01-01") & (df["Date"] < "2016-01-01")]
    test  = df[df["Date"] >= "2016-01-01"]
    return train, val, test

def build_preprocessor(num_cols):
    num = Pipeline([("fill", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, CAT_COLS)])