import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.train_model import stratified_patient_split

def main():
    feat_path = Path("models/feature_columns.json")
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    df = pd.read_csv("data/labeled_dataset.csv")
    train_df, val_df, test_df, _ = stratified_patient_split(df, seed=42)

    ep_df = pd.read_csv("data/episodes.csv")
    ep_df["onset_dt"] = pd.to_datetime(ep_df["onset_timestamp"])
    ep_df["end_dt"] = pd.to_datetime(ep_df["end_timestamp"])

    def create_mask(target_df):
        d = target_df.copy()
        d["dt"] = pd.to_datetime(d["timestamp"])
        mask = np.ones(len(d), dtype=bool)
        for _, ep in ep_df.iterrows():
            pid = ep["patient_id"]
            onset = ep["onset_dt"]
            end = ep["end_dt"] + pd.Timedelta(hours=72)
            m = (d["patient_id"] == pid) & (d["dt"] >= onset) & (d["dt"] < end)
            mask[m] = False
        return mask

    val_at_risk = create_mask(val_df)
    test_at_risk = create_mask(test_df)

    X_train = train_df[feature_cols]
    y_train = train_df["event_next_24h"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["event_next_24h"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["event_next_24h"].to_numpy()

    # Middle constraints
    middle_cst_dict = {c: 0 for c in feature_cols}
    for c in feature_cols:
        if c.startswith("pm25_"):
            middle_cst_dict[c] = 1
        elif c == "controller_adherence_7d":
            middle_cst_dict[c] = -1
        elif c.startswith("rescue_puffs_"):
            middle_cst_dict[c] = 1
        elif c.startswith("symptom_score_"):
            middle_cst_dict[c] = 1
        elif c in ["rr_deviation_from_baseline", "respiratory_rate_raw", "rr_rolling_mean_6h", "rr_rolling_mean_24h", "rr_recent_vs_usual"]:
            middle_cst_dict[c] = 1
        elif c in ["spo2_drop_from_baseline"]:
            middle_cst_dict[c] = 1
        elif c in ["spo2_raw", "spo2_mean_6h", "spo2_min_6h", "spo2_recent_vs_usual"]:
            middle_cst_dict[c] = -1
    middle_cst = [middle_cst_dict[c] for c in feature_cols]

    # Full constraints
    with open("models/monotonic_constraints.json", "r", encoding="utf-8") as f:
        full_dict = json.load(f)
    full_cst = [int(full_dict.get(c, 0)) for c in feature_cols]

    # RF
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)),
    ])
    rf_pipe.fit(X_train, y_train)

    # Full HGB
    full_hgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(monotonic_cst=full_cst, learning_rate=0.08, max_depth=8, max_iter=120, min_samples_leaf=50, l2_regularization=1.0, random_state=42)),
    ])
    full_hgb.fit(X_train, y_train)

    # Middle HGB
    mid_hgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(monotonic_cst=middle_cst, learning_rate=0.02, max_depth=None, max_leaf_nodes=31, max_iter=600, min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, n_iter_no_change=20, random_state=42)),
    ])
    mid_hgb.fit(X_train, y_train)

    val_rf = rf_pipe.predict_proba(X_val)[:, 1]
    val_full = full_hgb.predict_proba(X_val)[:, 1]
    val_mid = mid_hgb.predict_proba(X_val)[:, 1]

    test_rf = rf_pipe.predict_proba(X_test)[:, 1]
    test_full = full_hgb.predict_proba(X_test)[:, 1]
    test_mid = mid_hgb.predict_proba(X_test)[:, 1]

    print("ALL HOURS:")
    print(f"  Val:  RF={average_precision_score(y_val, val_rf):.4f}, FullHGB={average_precision_score(y_val, val_full):.4f}, MidHGB={average_precision_score(y_val, val_mid):.4f}")
    print(f"  Test: RF={average_precision_score(y_test, test_rf):.4f}, FullHGB={average_precision_score(y_test, test_full):.4f}, MidHGB={average_precision_score(y_test, test_mid):.4f}")

    print("\nAT RISK HOURS:")
    print(f"  Val:  RF={average_precision_score(y_val[val_at_risk], val_rf[val_at_risk]):.4f}, FullHGB={average_precision_score(y_val[val_at_risk], val_full[val_at_risk]):.4f}, MidHGB={average_precision_score(y_val[val_at_risk], val_mid[val_at_risk]):.4f}")
    print(f"  Test: RF={average_precision_score(y_test[test_at_risk], test_rf[test_at_risk]):.4f}, FullHGB={average_precision_score(y_test[test_at_risk], test_full[test_at_risk]):.4f}, MidHGB={average_precision_score(y_test[test_at_risk], test_mid[test_at_risk]):.4f}")

if __name__ == "__main__":
    main()
