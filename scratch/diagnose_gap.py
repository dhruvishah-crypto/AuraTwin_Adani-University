import json
import sys
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from src.train_model import stratified_patient_split

def main():
    df = pd.read_csv("data/labeled_dataset.csv")
    with open("models/feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    with open("models/monotonic_constraints.json", "r") as f:
        full_constraints_dict = json.load(f)

    train_df, val_df, test_df, _ = stratified_patient_split(df, seed=42)
    ep_df = pd.read_csv("data/episodes.csv")
    ep_df["onset_dt"] = pd.to_datetime(ep_df["onset_timestamp"])
    ep_df["end_dt"] = pd.to_datetime(ep_df["end_timestamp"])

    ep_by_pid = {}
    for pid, g in ep_df.groupby("patient_id"):
        ep_by_pid[pid] = [(row["onset_dt"], row["end_dt"] + pd.Timedelta(hours=90)) for _, row in g.iterrows()]

    def mark_at_risk(split_df):
        split_df = split_df.copy()
        split_df["dt"] = pd.to_datetime(split_df["timestamp"])
        at_risk_series = pd.Series(True, index=split_df.index)
        for pid, group_indices in split_df.groupby("patient_id").groups.items():
            if pid not in ep_by_pid:
                continue
            p_dts = split_df.loc[group_indices, "dt"]
            for onset, excl_end in ep_by_pid[pid]:
                mask = (p_dts >= onset) & (p_dts <= excl_end)
                at_risk_series.loc[group_indices[mask]] = False
        split_df["is_at_risk"] = at_risk_series
        return split_df

    print("Marking at-risk hours...", flush=True)
    train_df = mark_at_risk(train_df)
    val_df = mark_at_risk(val_df)
    test_df = mark_at_risk(test_df)

    X_train = train_df[feature_cols]
    y_train = train_df["event_next_24h"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["event_next_24h"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["event_next_24h"].to_numpy()

    val_at_risk_mask = val_df["is_at_risk"].to_numpy()
    test_at_risk_mask = test_df["is_at_risk"].to_numpy()

    print(f"Val hours: {len(val_df)}, At-risk: {val_at_risk_mask.sum()} ({val_at_risk_mask.mean()*100:.1f}%)")
    print(f"Test hours: {len(test_df)}, At-risk: {test_at_risk_mask.sum()} ({test_at_risk_mask.mean()*100:.1f}%)")

    # Train LR
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)

    # Train RF
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)),
    ])
    rf_pipe.fit(X_train, y_train)

    # Constrained HGB (Stage 4c operational)
    full_cst = [int(full_constraints_dict.get(c, 0)) for c in feature_cols]
    hgb_pipe_4c = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            monotonic_cst=full_cst,
            learning_rate=0.08,
            max_depth=6,
            max_iter=100,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])
    hgb_pipe_4c.fit(X_train, y_train)

    # Predict
    val_probs_lr = lr_pipe.predict_proba(X_val)[:, 1]
    val_probs_rf = rf_pipe.predict_proba(X_val)[:, 1]
    val_probs_hgb = hgb_pipe_4c.predict_proba(X_val)[:, 1]

    test_probs_lr = lr_pipe.predict_proba(X_test)[:, 1]
    test_probs_rf = rf_pipe.predict_proba(X_test)[:, 1]
    test_probs_hgb = hgb_pipe_4c.predict_proba(X_test)[:, 1]

    print("\n--- PR-AUC Comparison: All Hours vs At-Risk Hours ---")
    print("VALIDATION:")
    print(f"  LR:  All = {average_precision_score(y_val, val_probs_lr):.4f} | At-Risk = {average_precision_score(y_val[val_at_risk_mask], val_probs_lr[val_at_risk_mask]):.4f}")
    print(f"  RF:  All = {average_precision_score(y_val, val_probs_rf):.4f} | At-Risk = {average_precision_score(y_val[val_at_risk_mask], val_probs_rf[val_at_risk_mask]):.4f}")
    print(f"  HGB: All = {average_precision_score(y_val, val_probs_hgb):.4f} | At-Risk = {average_precision_score(y_val[val_at_risk_mask], val_probs_hgb[val_at_risk_mask]):.4f}")

    print("TEST:")
    print(f"  LR:  All = {average_precision_score(y_test, test_probs_lr):.4f} | At-Risk = {average_precision_score(y_test[test_at_risk_mask], test_probs_lr[test_at_risk_mask]):.4f}")
    print(f"  RF:  All = {average_precision_score(y_test, test_probs_rf):.4f} | At-Risk = {average_precision_score(y_test[test_at_risk_mask], test_probs_rf[test_at_risk_mask]):.4f}")
    print(f"  HGB: All = {average_precision_score(y_test, test_probs_hgb):.4f} | At-Risk = {average_precision_score(y_test[test_at_risk_mask], test_probs_hgb[test_at_risk_mask]):.4f}")

    # Item 5: Wider validation-only tuning pass
    print("\n--- Item 5: Wider Validation-Only Tuning Pass ---")
    # Relaxed constraints:
    # only constraints for pm25 (+1), adherence (-1), rescue (+1), symptoms (+1), spo2/rr deviation (+1)
    relaxed_cst_dict = {}
    for c in feature_cols:
        val = 0
        if c.startswith("pm25_"):
            val = 1
        elif c == "controller_adherence_7d":
            val = -1
        elif c.startswith("rescue_puffs_"):
            val = 1
        elif c.startswith("symptom_score_"):
            val = 1
        elif c in ["spo2_drop_from_baseline", "rr_deviation_from_baseline", "rr_recent_vs_usual"]:
            val = 1
        relaxed_cst_dict[c] = val

    relaxed_cst = [relaxed_cst_dict[c] for c in feature_cols]
    print(f"Full constraints non-zero: {sum(1 for c in full_cst if c != 0)}")
    print(f"Relaxed constraints non-zero: {sum(1 for c in relaxed_cst if c != 0)}")

    configs = [
        # Base learning rates down to 0.02, max_iter up to 600 with early stopping
        {"learning_rate": 0.02, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 20, "early_stopping": True},
        {"learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 30, "early_stopping": True},
        {"learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 15, "min_samples_leaf": 30, "early_stopping": True},
        {"learning_rate": 0.03, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 20, "early_stopping": True},
        {"learning_rate": 0.03, "max_iter": 500, "max_leaf_nodes": 20, "min_samples_leaf": 30, "early_stopping": True},
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20, "early_stopping": True},
        {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 15, "min_samples_leaf": 30, "early_stopping": True},
        {"learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 30, "early_stopping": True},
    ]

    print("\nTuning with FULL constraints:")
    best_full_val_pr = 0
    best_full_cfg = None
    for cfg in configs:
        hgb = HistGradientBoostingClassifier(
            monotonic_cst=full_cst,
            random_state=42,
            n_iter_no_change=15,
            validation_fraction=0.15,
            **cfg,
        )
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", hgb)])
        pipe.fit(X_train, y_train)
        val_pr = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])
        print(f"  Cfg {cfg} -> Val PR-AUC = {val_pr:.4f} (actual iters: {pipe.named_steps['clf'].n_iter_})")
        if val_pr > best_full_val_pr:
            best_full_val_pr = val_pr
            best_full_cfg = cfg

    print(f"\nBest FULL constraints: {best_full_cfg} -> Val PR-AUC = {best_full_val_pr:.4f}")

    print("\nTuning with RELAXED constraints:")
    best_rel_val_pr = 0
    best_rel_cfg = None
    for cfg in configs:
        hgb = HistGradientBoostingClassifier(
            monotonic_cst=relaxed_cst,
            random_state=42,
            n_iter_no_change=15,
            validation_fraction=0.15,
            **cfg,
        )
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", hgb)])
        pipe.fit(X_train, y_train)
        val_pr = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])
        print(f"  Cfg {cfg} -> Val PR-AUC = {val_pr:.4f} (actual iters: {pipe.named_steps['clf'].n_iter_})")
        if val_pr > best_rel_val_pr:
            best_rel_val_pr = val_pr
            best_rel_cfg = cfg

    print(f"\nBest RELAXED constraints: {best_rel_cfg} -> Val PR-AUC = {best_rel_val_pr:.4f}")

if __name__ == "__main__":
    main()
