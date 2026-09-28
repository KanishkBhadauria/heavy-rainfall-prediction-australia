import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)
from xgboost import XGBClassifier

from src.preprocessing import load_and_clean, time_split, build_preprocessor, TARGET
from src.features import add_features, FEATURE_COLS

def main():
    df = add_features(load_and_clean())
    train, val, _ = time_split(df)
    X_tr, y_tr = train.drop(columns=["Date", TARGET]), train[TARGET]
    X_va, y_va = val.drop(columns=["Date", TARGET]), val[TARGET]
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                               class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                 scale_pos_weight=spw, eval_metric="aucpr",
                                 n_jobs=-1, random_state=42),
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(FEATURE_COLS)), ("model", model)])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_va, pred),
                     "Precision": precision_score(y_va, pred, zero_division=0),
                     "Recall": recall_score(y_va, pred),
                     "F1": f1_score(y_va, pred),
                     "ROC_AUC": roc_auc_score(y_va, p),
                     "PR_AUC": average_precision_score(y_va, p)})
        joblib.dump(pipe, f"models/{name}.joblib")
        print(f"✅ {name} trained and saved")

    results = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).round(3)
    results.to_csv("reports/model_comparison.csv", index=False)
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()