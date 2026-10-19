import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.train_model import (
    stratified_patient_split,
    choose_decision_threshold,
    compute_metrics,
    compute_alert_burden,
    evaluate_alert_baselines,
    CalibratedModel,
)

def main():
    feat_path = Path('models/feature_columns.json')
    mono_path = Path('models/monotonic_constraints.json')
    with open(feat_path) as f:
        feature_cols = json.load(f)
    with open(mono_path) as f:
        cst = json.load(f)

    df = pd.read_csv('data/labeled_dataset.csv')
    train_df, val_df, test_df, _ = stratified_patient_split(df, seed=42)

    X_train, y_train = train_df[feature_cols], train_df['event_next_24h'].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df['event_next_24h'].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df['event_next_24h'].to_numpy()

    cst_list = [int(cst.get(c, 0)) for c in feature_cols]
    clf = HistGradientBoostingClassifier(
        monotonic_cst=cst_list,
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
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', clf)])
    pipe.fit(X_train, y_train)

    val_probs = pipe.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_probs, y_val)
    cal_model = CalibratedModel(pipe, calibrator)

    val_cal_probs = calibrator.predict(val_probs)
    thresh, meta = choose_decision_threshold(y_val, val_cal_probs, rule='maximize_f1')
    print(f"Threshold: {thresh:.4f}")

    test_cal_probs = cal_model.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, test_cal_probs, threshold=thresh)
    print(f"PR-AUC: {m['pr_auc']:.4f}")
    print(f"ROC-AUC: {m['roc_auc']:.4f}")
    print(f"Brier: {m['brier_score']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall: {m['recall']:.4f}")
    print(f"F1: {m['f1']:.4f}")
    print(f"CM: {m['confusion_matrix']}")

    burden = compute_alert_burden(test_df, test_cal_probs, threshold=thresh)
    print("\nAlert Burden:")
    print(f"Total alert episodes: {burden['total_alert_episodes']}")
    print(f"Anticipated: {burden['fraction_anticipated']*100:.2f}% ({burden['anticipated_onsets']}/{burden['total_true_onsets']})")
    print(f"Late: {burden['fraction_late_detection']*100:.2f}% ({burden['late_detection_onsets']}/{burden['total_true_onsets']})")
    print(f"Missed: {burden['fraction_missed']*100:.2f}% ({burden['missed_onsets']}/{burden['total_true_onsets']})")
    print(f"Median lead: {burden['lead_time_hours']['median']:.1f}h")
    print(f"False alerts/mo: {burden['false_alert_episodes_per_patient_month']:.2f}")

    # Alert precision & active/recovery breakdown
    print(f"Alert precision (onset within 24h): {burden.get('alert_precision')}")
    print(f"Alerts during active episode: {burden.get('alerts_during_active_episode')}")
    print(f"Alerts during recovery: {burden.get('alerts_during_recovery')}")
    print(f"Alerts during active or recovery: {burden.get('alerts_during_active_or_recovery')}")
    print(f"False alerts outside episodes: {burden.get('false_alerts_outside_episodes')}")

if __name__ == "__main__":
    main()
