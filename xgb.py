"""XGBoost training for a single domain.

Usage:
    python3 xgb.py "medicine"                  # single-pass (fast)
    python3 xgb.py "medicine" --tune           # hyperopt tuning
    python3 xgb.py "medicine" --tune --evals 200
"""

import argparse
import os
import sys

from lib import db, xgboost

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2.db")

# Feature labels for indices beyond the 768 embedding dims
EXTRA_FEATURE_NAMES = [
    "total_dims",
    "is_noun", "is_verb", "is_adj", "is_adv",
    "n_synsets", "n_domains",
    "centroid_cos", "knn_top5",
]


def feature_label(dim_id):
    if dim_id < 768:
        return f"dim_{dim_id}"
    idx = dim_id - 768
    if idx < len(EXTRA_FEATURE_NAMES):
        return EXTRA_FEATURE_NAMES[idx]
    return f"feat_{dim_id}"


def main():
    parser = argparse.ArgumentParser(description="XGBoost training for a domain")
    parser.add_argument("domain", help="Domain name (e.g. 'medicine')")
    parser.add_argument("--tune", action="store_true", help="Run hyperopt tuning (slow)")
    parser.add_argument("--evals", type=int, default=100, help="Number of hyperopt trials (default: 100)")
    args = parser.parse_args()

    con = db.get_connection(DB_PATH)

    print("Loading data...")
    means, stds = xgboost.load_dim_stats(con)
    emb_map = xgboost.load_embeddings(con)
    all_wids = list(emb_map.keys())
    group_map, n_groups, _ = xgboost.build_morphological_groups(con)
    total_dims_map = xgboost.load_total_dims(con)
    pos_flags_map = xgboost.load_pos_flags(con)
    word_meta_map = xgboost.load_word_meta(con)
    domains, skipped, excl_info = xgboost.load_domain_word_ids(
        con, min_words=20, exclude_sources={'morphy'}, exclude_stop=True
    )
    print(f"  {len(emb_map):,} embeddings, {n_groups:,} morph groups, {len(domains)} domains")
    print(f"  Filtering: {excl_info}")

    if args.domain not in domains:
        print(f"\nERROR: Domain '{args.domain}' not found or has < 20 matched words.")
        print(f"Available domains ({len(domains)}):")
        for d in sorted(domains.keys()):
            print(f"  {d} ({len(domains[d]):,})")
        con.close()
        sys.exit(1)

    pos_wids = domains[args.domain]
    print(f"\nDomain: {args.domain} ({len(pos_wids):,} positive word_ids)")

    result = xgboost.build_dataset(
        pos_wids, all_wids, emb_map, means, stds, group_map,
        total_dims_map=total_dims_map, pos_flags_map=pos_flags_map,
        word_meta_map=word_meta_map
    )
    if result is None:
        print("ERROR: Not enough data to build dataset.")
        con.close()
        sys.exit(1)

    X, y, groups, word_ids = result
    print(f"Dataset: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"  Positives: {int(y.sum()):,} | Negatives: {int(len(y) - y.sum()):,}")

    if args.tune:
        # Hyperopt tuning path
        print(f"\nTuning ({args.evals} trials)...")
        tune_result = xgboost.tune(X, y, groups, max_evals=args.evals)

        print(f"\n{'='*50}")
        print(f"BEST PARAMS")
        print(f"{'='*50}")
        for k, v in tune_result["best_params"].items():
            if isinstance(v, float):
                print(f"  {k:<20} {v:.6f}")
            else:
                print(f"  {k:<20} {v}")
        print(f"  {'best F1':<20} {tune_result['best_f1']:.4f}")
        print(f"  {'tuning time':<20} {tune_result['elapsed_s']:.1f}s")

        # Retrain with best params for full metrics
        print(f"\nRetraining with best params...")
        final = xgboost.train(X, y, groups, **tune_result["best_params"])
    else:
        # Single-pass with default params
        print(f"\nTraining (single pass)...")
        final = xgboost.train(X, y, groups)

    if final is None:
        print("ERROR: Training failed.")
        con.close()
        sys.exit(1)

    m = final["metrics"]
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  AUC-PR:    {m['auc_pr']:.4f}")
    print(f"  Time:      {final['elapsed_s']:.2f}s")

    # Top features
    top = xgboost.top_features(final["importances"], n=15)
    print(f"\nTop 15 features:")
    for f in top:
        label = feature_label(f["dim_id"])
        print(f"  {label:<14} importance={f['importance']:.4f}")

    con.close()


if __name__ == "__main__":
    main()
