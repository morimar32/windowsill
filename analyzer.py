import numpy as np
from scipy import stats
from tqdm import tqdm

import config
import database


def compute_basic_stats(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min_val": float(np.min(values)),
        "max_val": float(np.max(values)),
        "median": float(np.median(values)),
        "skewness": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
    }


def zscore_threshold(mean, std, n_sigma=None):
    if n_sigma is None:
        n_sigma = config.ZSCORE_THRESHOLD
    return mean + n_sigma * std


def identify_members(dim_id, values, word_ids, threshold, mean, std):
    members = []
    for val, wid in zip(values, word_ids):
        if val >= threshold:
            z = (val - mean) / std if std > 0 else 0.0
            members.append((dim_id, wid, float(val), float(z)))
    return members


def analyze_dimension(dim_id, values, word_ids):
    basic = compute_basic_stats(values)
    threshold = zscore_threshold(basic["mean"], basic["std"])

    members = identify_members(dim_id, values, word_ids, threshold, basic["mean"], basic["std"])
    n_members = len(members)
    selectivity = n_members / len(values) if len(values) > 0 else 0.0

    stats_dict = {
        "dim_id": dim_id,
        "mean": basic["mean"],
        "std": basic["std"],
        "min_val": basic["min_val"],
        "max_val": basic["max_val"],
        "median": basic["median"],
        "skewness": basic["skewness"],
        "kurtosis": basic["kurtosis"],
        "threshold": float(threshold),
        "threshold_method": "zscore",
        "n_members": n_members,
        "selectivity": selectivity,
    }

    return stats_dict, members


def run_analysis(con, embedding_matrix=None, word_ids=None):
    if embedding_matrix is None or word_ids is None:
        print("Loading embeddings from database...")
        embedding_matrix, word_ids = database.load_embedding_matrix(con)

    n_words, n_dims = embedding_matrix.shape
    print(f"Analyzing {n_dims} dimensions across {n_words} words...")

    con.execute("DELETE FROM dim_stats")
    con.execute("DELETE FROM dim_memberships")

    all_stats = []
    all_memberships = []

    for dim_id in tqdm(range(n_dims), desc="Analyzing dimensions"):
        values = embedding_matrix[:, dim_id].astype(np.float64)
        stats_dict, members = analyze_dimension(dim_id, values, word_ids)
        all_stats.append(stats_dict)
        all_memberships.extend(members)

        if (dim_id + 1) % config.COMMIT_INTERVAL == 0:
            database.insert_dim_stats(con, all_stats)
            database.insert_dim_memberships(con, all_memberships)
            all_stats = []
            all_memberships = []

    if all_stats:
        database.insert_dim_stats(con, all_stats)
    if all_memberships:
        database.insert_dim_memberships(con, all_memberships)

    total_memberships = con.execute("SELECT COUNT(*) FROM dim_memberships").fetchone()[0]
    avg_sel = con.execute("SELECT AVG(selectivity) FROM dim_stats").fetchone()[0]

    print(f"\nAnalysis complete:")
    print(f"  Threshold method: zscore (z={config.ZSCORE_THRESHOLD})")
    print(f"  Total memberships: {total_memberships:,}")
    print(f"  Avg selectivity: {avg_sel:.4f}" if avg_sel else "")


def run_sense_analysis(con, sense_embeddings, sense_ids):
    dim_stats_rows = con.execute(
        "SELECT dim_id, threshold, mean, std FROM dim_stats ORDER BY dim_id"
    ).fetchall()

    n_senses, n_dims = sense_embeddings.shape
    print(f"Analyzing {n_senses} senses against {n_dims} existing dimension thresholds...")

    con.execute("DELETE FROM sense_dim_memberships")

    all_memberships = []
    for dim_id, threshold, mean, std in tqdm(dim_stats_rows, desc="Sense analysis"):
        values = sense_embeddings[:, dim_id].astype(np.float64)
        for val, sid in zip(values, sense_ids):
            if val >= threshold:
                z = (val - mean) / std if std > 0 else 0.0
                all_memberships.append((dim_id, sid, float(val), float(z)))

        if len(all_memberships) >= 10000:
            database.insert_sense_dim_memberships(con, all_memberships)
            all_memberships = []

    if all_memberships:
        database.insert_sense_dim_memberships(con, all_memberships)

    # Update total_dims on word_senses
    con.execute("""
        UPDATE word_senses SET total_dims = (
            SELECT COUNT(*) FROM sense_dim_memberships sdm
            WHERE sdm.sense_id = word_senses.sense_id
        )
    """)

    total = con.execute("SELECT COUNT(*) FROM sense_dim_memberships").fetchone()[0]
    print(f"  Sense analysis complete: {total:,} sense-dimension memberships")
