import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline

from src.train_model import (
    stratified_patient_split,
    choose_decision_threshold,
    compute_metrics,
    compute_patient_bootstrap_ci,
    run_ablation_study,
    plot_diagnostic_curves,
    write_model_card,
    update_readme_metrics,
    CalibratedModel,
    WORKSPACE_ROOT,
    MANDATORY_DISCLAIMER,
)
from src.digital_twin import DigitalTwin

def main():
    data_path = Path("data/labeled_dataset.csv")
    feat_path = Path("models/feature_columns.json")
    mono_path = Path("models/monotonic_constraints.json")
    
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(mono_path, "r", encoding="utf-8") as f:
        constraints_dict = json.load(f)
        
    df = pd.read_csv(data_path)
    train_df, val_df, test_df, split_summary = stratified_patient_split(df, seed=42)
    
    X_train = train_df[feature_cols]
    y_train = train_df["event_next_24h"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["event_next_24h"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["event_next_24h"].to_numpy()

    # Baselines: LR and RF
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)

    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42)),
    ])
    rf_pipe.fit(X_train, y_train)

    # Operational Middle Constrained HGB
    hgb_monotonic_cst = [int(constraints_dict.get(c, 0)) for c in feature_cols]
    
    cand_clf = HistGradientBoostingClassifier(
        monotonic_cst=hgb_monotonic_cst,
        learning_rate=0.02,
        max_depth=None,
        max_leaf_nodes=31,
        max_iter=600,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    hgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", cand_clf),
    ])
    hgb_pipe.fit(X_train, y_train)

    # Precompute feature importances
    clf_hgb = hgb_pipe.named_steps["clf"]
    if hasattr(clf_hgb, "_predictors"):
        n_feat = len(feature_cols)
        importances = np.zeros(n_feat)
        for predictor_list in clf_hgb._predictors:
            for tree in predictor_list:
                nodes = tree.nodes
                gain_mask = nodes["gain"] > 0
                np.add.at(importances, nodes["feature_idx"][gain_mask], nodes["gain"][gain_mask])
        tot = np.sum(importances)
        if tot > 0:
            importances /= tot
        clf_hgb.feature_importances_ = importances

    val_probs_lr = lr_pipe.predict_proba(X_val)[:, 1]
    val_probs_rf = rf_pipe.predict_proba(X_val)[:, 1]
    val_probs_hgb = hgb_pipe.predict_proba(X_val)[:, 1]

    val_pr_lr = float(average_precision_score(y_val, val_probs_lr))
    val_pr_rf = float(average_precision_score(y_val, val_probs_rf))
    val_pr_hgb = float(average_precision_score(y_val, val_probs_hgb))

    test_probs_lr = lr_pipe.predict_proba(X_test)[:, 1]
    test_probs_rf = rf_pipe.predict_proba(X_test)[:, 1]
    test_probs_hgb = hgb_pipe.predict_proba(X_test)[:, 1]

    print(f"Validation PR-AUC: LR={val_pr_lr:.4f}, RF={val_pr_rf:.4f}, HGB={val_pr_hgb:.4f}")

    # Calibration on validation only
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_hgb, y_val)
    calibrated_model = CalibratedModel(hgb_pipe, calibrator, method="isotonic")

    val_cal_probs = calibrator.predict(val_probs_hgb)
    decision_threshold, threshold_meta = choose_decision_threshold(y_val, val_cal_probs, rule="maximize_f1")
    print(f"Selected Decision Threshold on Val: {decision_threshold:.4f} ({threshold_meta['rule_description']})")

    threshold_low = float(np.percentile(val_cal_probs, 75))
    threshold_high = float(np.percentile(val_cal_probs, 90))
    print(f"Bands: Low < {threshold_low:.4f}, Watch: [{threshold_low:.4f}, {threshold_high:.4f}), High >= {threshold_high:.4f}")

    test_cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
    raw_test_metrics = compute_metrics(y_test, test_probs_hgb, threshold=decision_threshold)
    cal_test_metrics = compute_metrics(y_test, test_cal_probs, threshold=decision_threshold)

    low_mask = test_cal_probs < threshold_low
    watch_mask = (test_cal_probs >= threshold_low) & (test_cal_probs < threshold_high)
    high_mask = test_cal_probs >= threshold_high

    obs_rate_low = float(np.mean(y_test[low_mask])) if np.sum(low_mask) > 0 else 0.0
    obs_rate_watch = float(np.mean(y_test[watch_mask])) if np.sum(watch_mask) > 0 else 0.0
    obs_rate_high = float(np.mean(y_test[high_mask])) if np.sum(high_mask) > 0 else 0.0

    print(f"Observed rates: Low={obs_rate_low:.4f}, Watch={obs_rate_watch:.4f}, High={obs_rate_high:.4f}")
    assert obs_rate_low < obs_rate_watch < obs_rate_high, "Monotonicity of demo bands violated!"
    print("Demo bands monotonicity verified: Low < Watch < High")

if __name__ == "__main__":
    main()
