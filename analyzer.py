import warnings
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
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


def _convergence_warning():
    try:
        from sklearn.exceptions import ConvergenceWarning
        return ConvergenceWarning
    except ImportError:
        return UserWarning


def _fit_gmm(values, n_components, cfg):
    X = values.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=_convergence_warning())
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=cfg.GMM_RANDOM_STATE,
                max_iter=cfg.GMM_MAX_ITER,
                reg_covar=cfg.GMM_REG_COVAR,
            )
            gmm.fit(X)
            return gmm, gmm.bic(X)
        except Exception:
            return None, None


def test_bimodality_bic(values, cfg=None):
    """Compare 1-component vs 2-component GMM using BIC."""
    if cfg is None:
        cfg = config

    gmm1, bic_1 = _fit_gmm(values, 1, cfg)
    if gmm1 is None:
        return False, None, None, None

    gmm2, bic_2 = _fit_gmm(values, 2, cfg)
    if gmm2 is None:
        return False, float(bic_1), None, None

    bic_delta = bic_1 - bic_2  # positive = 2-component is better
    is_bimodal = bic_delta > cfg.GMM_BIC_THRESHOLD

    if not is_bimodal:
        return False, float(bic_1), float(bic_2), None

    # Reject if minor component is < 1% of the data
    weights = gmm2.weights_.flatten()
    if min(weights) < 0.01:
        return False, float(bic_1), float(bic_2), None

    means = gmm2.means_.flatten()
    stds = np.sqrt(gmm2.covariances_.flatten())
    idx_low, idx_high = np.argmin(means), np.argmax(means)
    mean_low, mean_high = means[idx_low], means[idx_high]
    std_low, std_high = stds[idx_low], stds[idx_high]

    if abs(mean_high - mean_low) < 1e-6 or std_low <= 0 or std_high <= 0:
        return False, float(bic_1), float(bic_2), None

    threshold = compute_gmm_intersection(mean_low, std_low, mean_high, std_high)

    gmm_result = {
        "threshold": threshold,
        "gmm_mean_low": float(mean_low),
        "gmm_mean_high": float(mean_high),
        "gmm_std_low": float(std_low),
        "gmm_std_high": float(std_high),
    }
    return True, float(bic_1), float(bic_2), gmm_result


def compute_gmm_intersection(mean_low, std_low, mean_high, std_high):
    return (mean_low * std_high + mean_high * std_low) / (std_low + std_high)


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
    is_bimodal, bic_1, bic_2, gmm_result = test_bimodality_bic(values)

    threshold_method = "zscore"
    threshold = zscore_threshold(basic["mean"], basic["std"])
    gmm_mean_low = gmm_mean_high = gmm_std_low = gmm_std_high = None

    if is_bimodal and gmm_result is not None:
        threshold = gmm_result["threshold"]
        threshold_method = "gmm"
        gmm_mean_low = gmm_result["gmm_mean_low"]
        gmm_mean_high = gmm_result["gmm_mean_high"]
        gmm_std_low = gmm_result["gmm_std_low"]
        gmm_std_high = gmm_result["gmm_std_high"]

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
        "threshold_method": threshold_method,
        "is_bimodal": is_bimodal,
        "bic_1": bic_1,
        "bic_2": bic_2,
        "gmm_mean_low": gmm_mean_low,
        "gmm_mean_high": gmm_mean_high,
        "gmm_std_low": gmm_std_low,
        "gmm_std_high": gmm_std_high,
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

    bimodal_count = con.execute("SELECT COUNT(*) FROM dim_stats WHERE is_bimodal").fetchone()[0]
    gmm_count = con.execute("SELECT COUNT(*) FROM dim_stats WHERE threshold_method = 'gmm'").fetchone()[0]
    total_memberships = con.execute("SELECT COUNT(*) FROM dim_memberships").fetchone()[0]

    print(f"\nAnalysis complete:")
    print(f"  Bimodal dimensions: {bimodal_count}")
    print(f"  GMM thresholds: {gmm_count}")
    print(f"  Z-score thresholds: {n_dims - gmm_count}")
    print(f"  Total memberships: {total_memberships}")


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
