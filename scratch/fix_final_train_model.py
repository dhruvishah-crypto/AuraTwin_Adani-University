from pathlib import Path

train_model_path = Path("src/train_model.py")
content = train_model_path.read_text(encoding="utf-8")

# 1. Update line 2140 where constraints_dict was used
content = content.replace("full_constraints_dict=constraints_dict,", "full_constraints_dict=middle_constraints,")

# 2. In tune_constrained_hgb_wider, streamline grid to save redundant training time
old_grid = '''    grid = [
        {"learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.03, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.08, "max_iter": 600, "max_leaf_nodes": 63, "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]'''

new_grid = '''    # Evaluated candidates from validated grid search space
    grid = [
        {"learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]'''

if old_grid in content:
    content = content.replace(old_grid, new_grid)
    print("Streamlined wider tuning grid.")

# Also ensure full_monotonic_cst uses the 52-feature full constraint benchmark
old_full_cst = "full_monotonic_cst = [int(full_constraints_dict.get(c, 0)) for c in feature_cols]"
new_full_cst = """full_monotonic_cst = [1 if (c.startswith("pm25_") or c.startswith("rescue_") or c.startswith("symptom_") or c in ["rr_deviation_from_baseline", "spo2_drop_from_baseline", "respiratory_rate_raw", "rr_rolling_mean_6h", "rr_rolling_mean_24h", "rr_recent_vs_usual", "rr_rolling_std_24h", "sleep_deficit"]) else (-1 if (c in ["controller_adherence_7d", "spo2_raw", "spo2_mean_6h", "spo2_min_6h", "spo2_mean_24h", "spo2_min_24h", "spo2_recent_vs_usual"]) else 0) for c in feature_cols]"""

if old_full_cst in content:
    content = content.replace(old_full_cst, new_full_cst)
    print("Updated full_monotonic_cst in tune_constrained_hgb_wider.")

train_model_path.write_text(content, encoding="utf-8")
print("Saved final fixes to train_model.py")
