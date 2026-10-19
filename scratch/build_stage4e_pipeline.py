import re
from pathlib import Path

train_model_path = Path("src/train_model.py")
content = train_model_path.read_text(encoding="utf-8")

# 1. Update diagnose_rf_hgb_gap to support both Stage 4c benchmark and Stage 4e operational middle model
old_diagnose_start = "def diagnose_rf_hgb_gap("
old_diagnose_body = """    val_probs_lr = lr_pipe.predict_proba(X_val)[:, 1]
    val_probs_rf = rf_pipe.predict_proba(X_val)[:, 1]
    val_probs_hgb = hgb_pipe.predict_proba(X_val)[:, 1]

    test_probs_lr = lr_pipe.predict_proba(X_test)[:, 1]
    test_probs_rf = rf_pipe.predict_proba(X_test)[:, 1]
    test_probs_hgb = hgb_pipe.predict_proba(X_test)[:, 1]"""

new_diagnose_body = """    val_probs_lr = lr_pipe.predict_proba(X_val)[:, 1]
    val_probs_rf = rf_pipe.predict_proba(X_val)[:, 1]
    val_probs_hgb = hgb_pipe.predict_proba(X_val)[:, 1]

    test_probs_lr = lr_pipe.predict_proba(X_test)[:, 1]
    test_probs_rf = rf_pipe.predict_proba(X_test)[:, 1]
    test_probs_hgb = hgb_pipe.predict_proba(X_test)[:, 1]

    # Benchmark Stage 4c full-constrained HGB for gap reduction continuity
    full_cst = [1 if (c.startswith("pm25_") or c.startswith("rescue_") or c.startswith("symptom_") or c in ["rr_deviation_from_baseline", "spo2_drop_from_baseline", "respiratory_rate_raw", "rr_rolling_mean_6h", "rr_rolling_mean_24h", "rr_recent_vs_usual", "rr_rolling_std_24h", "sleep_deficit"]) else (-1 if (c in ["controller_adherence_7d", "spo2_raw", "spo2_mean_6h", "spo2_min_6h", "spo2_mean_24h", "spo2_min_24h", "spo2_recent_vs_usual"]) else 0) for c in feature_cols]
    bench_clf = HistGradientBoostingClassifier(monotonic_cst=full_cst, learning_rate=0.08, max_depth=8, max_iter=120, min_samples_leaf=50, l2_regularization=1.0, random_state=42)
    bench_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", bench_clf)])
    bench_pipe.fit(train_df[feature_cols], train_df["event_next_24h"].to_numpy())
    val_probs_bench = bench_pipe.predict_proba(X_val)[:, 1]
    test_probs_bench = bench_pipe.predict_proba(X_test)[:, 1]"""

if old_diagnose_body in content:
    content = content.replace(old_diagnose_body, new_diagnose_body)
    print("Updated diagnose_rf_hgb_gap body.")

# Update the val_pr_all and test_pr_all in diagnose_rf_hgb_gap
old_val_pr_dict = """    val_pr_all = {
        "logistic_regression": round(float(average_precision_score(y_val, val_probs_lr)), 4),
        "random_forest": round(float(average_precision_score(y_val, val_probs_rf)), 4),
        "constrained_hgb": round(float(average_precision_score(y_val, val_probs_hgb)), 4),
    }
    test_pr_all = {
        "logistic_regression": round(float(average_precision_score(y_test, test_probs_lr)), 4),
        "random_forest": round(float(average_precision_score(y_test, test_probs_rf)), 4),
        "constrained_hgb": round(float(average_precision_score(y_test, test_probs_hgb)), 4),
    }

    val_pr_at_risk = {
        "logistic_regression": round(float(average_precision_score(y_val[val_at_risk], val_probs_lr[val_at_risk])), 4),
        "random_forest": round(float(average_precision_score(y_val[val_at_risk], val_probs_rf[val_at_risk])), 4),
        "constrained_hgb": round(float(average_precision_score(y_val[val_at_risk], val_probs_hgb[val_at_risk])), 4),
    }
    test_pr_at_risk = {
        "logistic_regression": round(float(average_precision_score(y_test[test_at_risk], test_probs_lr[test_at_risk])), 4),
        "random_forest": round(float(average_precision_score(y_test[test_at_risk], test_probs_rf[test_at_risk])), 4),
        "constrained_hgb": round(float(average_precision_score(y_test[test_at_risk], test_probs_hgb[test_at_risk])), 4),
    }"""

new_val_pr_dict = """    val_pr_all = {
        "logistic_regression": round(float(average_precision_score(y_val, val_probs_lr)), 4),
        "random_forest": round(float(average_precision_score(y_val, val_probs_rf)), 4),
        "constrained_hgb": round(float(average_precision_score(y_val, val_probs_bench)), 4),
        "middle_constrained_hgb": round(float(average_precision_score(y_val, val_probs_hgb)), 4),
    }
    test_pr_all = {
        "logistic_regression": round(float(average_precision_score(y_test, test_probs_lr)), 4),
        "random_forest": round(float(average_precision_score(y_test, test_probs_rf)), 4),
        "constrained_hgb": round(float(average_precision_score(y_test, test_probs_bench)), 4),
        "middle_constrained_hgb": round(float(average_precision_score(y_test, test_probs_hgb)), 4),
    }

    val_pr_at_risk = {
        "logistic_regression": round(float(average_precision_score(y_val[val_at_risk], val_probs_lr[val_at_risk])), 4),
        "random_forest": round(float(average_precision_score(y_val[val_at_risk], val_probs_rf[val_at_risk])), 4),
        "constrained_hgb": round(float(average_precision_score(y_val[val_at_risk], val_probs_bench[val_at_risk])), 4),
        "middle_constrained_hgb": round(float(average_precision_score(y_val[val_at_risk], val_probs_hgb[val_at_risk])), 4),
    }
    test_pr_at_risk = {
        "logistic_regression": round(float(average_precision_score(y_test[test_at_risk], test_probs_lr[test_at_risk])), 4),
        "random_forest": round(float(average_precision_score(y_test[test_at_risk], test_probs_rf[test_at_risk])), 4),
        "constrained_hgb": round(float(average_precision_score(y_test[test_at_risk], test_probs_bench[test_at_risk])), 4),
        "middle_constrained_hgb": round(float(average_precision_score(y_test[test_at_risk], test_probs_hgb[test_at_risk])), 4),
    }"""

if old_val_pr_dict in content:
    content = content.replace(old_val_pr_dict, new_val_pr_dict)
    print("Updated val_pr_all and val_pr_at_risk dictionaries.")

train_model_path.write_text(content, encoding="utf-8")
print("Saved updated train_model.py")
