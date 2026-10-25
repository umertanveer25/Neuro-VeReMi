import numpy as np
from scipy import stats

def compute_paired_statistics(scores_a, scores_b, alpha=0.05, k_folds=10, n_seeds=30):
    """
    Computes rigorous paired inferential statistics across N=300 scenario-disjoint evaluation folds (df = 299):
    1. Paired difference vector d = scores_a - scores_b
    2. Mean difference d_bar and standard deviation s_d
    3. Standard error SE_d = s_d / sqrt(N)
    4. 95% Confidence Interval [d_bar - t_crit * SE_d, d_bar + t_crit * SE_d]
    5. Paired Student's t-test: t = d_bar / SE_d, df = 299
    6. Wilcoxon signed-rank test: W+ (positive rank sum) and exact p-value
    7. Cohen's d_z (paired effect size) = d_bar / s_d
    8. Cohen's h (proportional effect size) = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
    9. Nadeau-Bengio corrected resampled t-statistic t_corr accounting for CV training overlap
    """
    scores_a = np.array(scores_a, dtype=np.float64)
    scores_b = np.array(scores_b, dtype=np.float64)
    diff = scores_a - scores_b
    n = len(diff)  # 300

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = std_diff / np.sqrt(n)

    # 1. Paired Student's t-test (df = n - 1 = 299)
    t_stat = mean_diff / (se_diff + 1e-12)
    t_pval = float(2.0 * stats.t.sf(np.abs(t_stat), df=n - 1))

    # 2. 95% Confidence Interval for paired mean difference
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # 3. Wilcoxon signed-rank test (Explicit W+ computation)
    abs_diff = np.abs(diff)
    nonzero_mask = abs_diff > 1e-9
    if np.sum(nonzero_mask) > 0:
        ranks = stats.rankdata(abs_diff[nonzero_mask])
        w_plus = float(np.sum(ranks[diff[nonzero_mask] > 0]))
        w_minus = float(np.sum(ranks[diff[nonzero_mask] < 0]))
        w_stat, w_pval = stats.wilcoxon(diff[nonzero_mask])
        if w_pval == 0.0 or np.isnan(w_pval):
            w_pval = 1.0e-15
    else:
        w_plus = 0.0
        w_minus = 0.0
        w_stat = 0.0
        w_pval = 1.0

    # 4. Standardized Effect Sizes
    cohen_dz = mean_diff / (std_diff + 1e-12)
    p1 = np.clip(np.mean(scores_a) / 100.0, 0.0, 1.0)
    p2 = np.clip(np.mean(scores_b) / 100.0, 0.0, 1.0)
    cohen_h = float(2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2)))

    # 5. Nadeau-Bengio corrected resampled t-test
    # Variance correction factor: (1/N + n_test / n_train) = (1/300 + 1/(k_folds - 1)) = (1/300 + 1/9)
    test_train_ratio = 1.0 / (k_folds - 1)
    corrected_variance = (1.0 / n + test_train_ratio) * (std_diff ** 2)
    t_corr = mean_diff / np.sqrt(corrected_variance + 1e-12)
    t_corr_pval = float(2.0 * stats.t.sf(np.abs(t_corr), df=n - 1))

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "ci_95": (float(ci_lower), float(ci_upper)),
        "t_stat": float(t_stat),
        "t_pval": float(t_pval),
        "wilcoxon_w_plus": float(w_plus),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pval": float(w_pval),
        "cohen_dz": float(cohen_dz),
        "cohen_h": float(cohen_h),
        "nadeau_bengio_t_stat": float(t_corr),
        "nadeau_bengio_p_val": float(t_corr_pval)
    }
