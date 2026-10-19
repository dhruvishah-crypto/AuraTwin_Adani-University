"""Adherence diagnosis for Stage 6b.

Computes:
1. Distribution of controller_adherence_7d in train and test splits.
2. Permutation importance on the validation split.
3. Number of tree splits using controller_adherence_7d in the operational model.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.train_model import stratified_patient_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    data_path = WORKSPACE_ROOT / "data" / "labeled_dataset.csv"
    feat_cols_path = WORKSPACE_ROOT / "models" / "feature_columns.json"
    model_path = WORKSPACE_ROOT / "models" / "model.joblib"

    with open(feat_cols_path, "r") as f:
        feature_cols: list[str] = json.load(f)

    model = joblib.load(model_path)
    # Extract base pipeline/classifier
    base_pipe = getattr(model, "base_pipeline", model)
    clf = base_pipe.named_steps.get("clf", base_pipe)

    df = pd.read_csv(data_path)
    train_df, val_df, test_df, _ = stratified_patient_split(df, seed=42)

    adh_col = "controller_adherence_7d"

    # 1. Distribution in train and test
    train_adh = train_df[adh_col]
    test_adh = test_df[adh_col]
    val_adh = val_df[adh_col]

    def stats_dict(s: pd.Series) -> dict:
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "frac_full_1_0": float((s >= 0.999).mean()),
            "frac_below_0_8": float((s < 0.8).mean()),
            "frac_below_0_5": float((s < 0.5).mean()),
        }

    train_stats = stats_dict(train_adh)
    test_stats = stats_dict(test_adh)
    val_stats = stats_dict(val_adh)

    print("=== 1. DISTRIBUTION OF CONTROLLER_ADHERENCE_7D ===")
    print("TRAIN STATS:", json.dumps(train_stats, indent=2))
    print("TEST STATS:", json.dumps(test_stats, indent=2))
    print("VAL STATS:", json.dumps(val_stats, indent=2))

    # 2. Number of tree splits in the operational model using controller_adherence_7d
    adh_idx = feature_cols.index(adh_col)
    total_trees = 0
    total_nodes = 0
    splits_on_adh = 0
    feature_split_counts = {c: 0 for c in feature_cols}

    if hasattr(clf, "_predictors"):
        for predictor_list in clf._predictors:
            for tree in predictor_list:
                total_trees += 1
                nodes = tree.nodes
                # Internal nodes have gain > 0 and feature_idx >= 0
                internal_mask = nodes["gain"] > 0
                total_nodes += int(internal_mask.sum())
                f_indices = nodes["feature_idx"][internal_mask]
                for f_i in f_indices:
                    feature_split_counts[feature_cols[f_i]] += 1
                splits_on_adh += int((f_indices == adh_idx).sum())

    print("\n=== 2. TREE SPLIT COUNT ===")
    print(f"Total boosting trees: {total_trees}")
    print(f"Total internal split nodes across all trees: {total_nodes}")
    print(f"Splits using '{adh_col}' (idx {adh_idx}): {splits_on_adh} ({splits_on_adh / max(1, total_nodes)*100:.3f}% of all splits)")

    # Top 10 most split features
    top_split_feats = sorted(feature_split_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop 10 Most Split Features across trees:")
    for f_name, count in top_split_feats:
        print(f"  {f_name:<30}: {count} splits")

    # 3. Permutation importance on validation split
    print("\n=== 3. PERMUTATION IMPORTANCE ON VALIDATION SPLIT ===")
    from sklearn.metrics import average_precision_score

    X_val = val_df[feature_cols].copy()
    y_val = val_df["event_next_24h"].to_numpy()

    base_probs = model.predict_proba(X_val)[:, 1]
    base_pr_auc = float(average_precision_score(y_val, base_probs))
    print(f"Base Validation PR-AUC: {base_pr_auc:.6f}")

    rng = np.random.default_rng(42)
    n_repeats = 5
    perm_importances = {}

    for i, col in enumerate(feature_cols):
        drops = []
        for r in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[col] = rng.permutation(X_val[col].to_numpy())
            perm_probs = model.predict_proba(X_perm)[:, 1]
            perm_pr = float(average_precision_score(y_val, perm_probs))
            drops.append(base_pr_auc - perm_pr)
        perm_importances[col] = {
            "mean": float(np.mean(drops)),
            "std": float(np.std(drops)),
        }
        if col == adh_col or (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(feature_cols)} features... ({col}: mean drop = {perm_importances[col]['mean']:.6f})")

    sorted_perm = sorted(perm_importances.items(), key=lambda kv: kv[1]["mean"], reverse=True)

    print(f"Permutation PR-AUC Importance for '{adh_col}':")
    print(f"  Mean Drop: {perm_importances[adh_col]['mean']:.6f} +/- {perm_importances[adh_col]['std']:.6f}")
    
    # Find rank of adherence
    adh_rank = [k for k, _ in sorted_perm].index(adh_col) + 1
    print(f"  Rank: {adh_rank} / {len(feature_cols)}")

    print("\nTop 10 Features by Permutation Importance:")
    for col, imp in sorted_perm[:10]:
        print(f"  {col:<30}: {imp['mean']:+.6f} +/- {imp['std']:.6f}")

    # Also bottom 5 features
    print("\nBottom 5 Features by Permutation Importance:")
    for col, imp in sorted_perm[-5:]:
        print(f"  {col:<30}: {imp['mean']:+.6f} +/- {imp['std']:.6f}")

    diagnosis_results = {
        "train_stats": train_stats,
        "test_stats": test_stats,
        "val_stats": val_stats,
        "splits_on_adherence": splits_on_adh,
        "total_tree_splits": total_nodes,
        "total_trees": total_trees,
        "adherence_split_fraction": splits_on_adh / max(1, total_nodes),
        "permutation_importance": perm_importances[adh_col],
        "adherence_rank": adh_rank,
        "top_split_features": top_split_feats,
        "top_permutation_features": sorted_perm[:10],
    }

    out_diag = WORKSPACE_ROOT / "models" / "adherence_diagnosis.json"
    with open(out_diag, "w") as f:
        json.dump(diagnosis_results, f, indent=2)
    print(f"\nSaved diagnosis to {out_diag}")

if __name__ == "__main__":
    main()
