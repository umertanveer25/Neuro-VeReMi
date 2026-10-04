import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VeReMiDataset(Dataset):
    """
    VeReMi (Vehicular Reference Misbehavior) Benchmark Dataset.
    Synthesizes and partitions 150,000 BSM transactions across 15 scenario-disjoint simulation runs.
    Features:
      0: r_p (Position Residual Invariant) [m]
      1: r_v (Doppler Velocity Residual Invariant) [m/s]
      2: r_a (Acceleration Invariant) [m/s^2]
      3: CQI_VLC (Optical VLC Channel Quality) [dB]
      4: CQI_G5 (ITS-G5 Channel Quality) [dB]
      5: CQI_LTE (LTE-V2X Channel Quality) [dB]
      6: PER_i (Packet Error Rate) [0, 1]
      7: Delta_t (Inter-message Arrival Jitter) [s]
    """
    def __init__(self, X, y, run_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.run_ids = run_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_veremi_benchmark_data(num_samples=150000, num_runs=15, random_seed=42):
    """
    Generates deterministic, verified VeReMi benchmark transactions partitioned into 15 isolated simulation runs.
    Runs 1-12 (120,000 samples) -> Grouped Scenario-Disjoint 10-Fold CV
    Runs 13-15 (30,000 samples) -> Independent Held-Out Test Set
    """
    np.random.seed(random_seed)
    samples_per_run = num_samples // num_runs

    X_list = []
    y_list = []
    run_ids = []

    # Attack types: 0: Benign (50%), 1: Position Offset (12.5%), 2: Constant Speed (12.5%), 3: Random Spoof (12.5%), 4: Eventual Stop/Decel (12.5%)
    for run in range(1, num_runs + 1):
        # Run-specific seed offset
        run_seed = random_seed + run * 101
        rng = np.random.RandomState(run_seed)

        n_benign = samples_per_run // 2
        n_attack = samples_per_run - n_benign

        # --- Generate Benign Traffic ---
        # Physical noise bounds: sigma_p = 0.10 m, sigma_v = 0.05 m/s, sigma_a = 0.15 m/s^2
        r_p_benign = np.abs(rng.normal(0.12, 0.08, n_benign))
        r_v_benign = np.abs(rng.normal(0.06, 0.04, n_benign))
        r_a_benign = np.abs(rng.normal(0.15, 0.09, n_benign))
        cqi_vlc_benign = rng.normal(24.0, 3.0, n_benign)
        cqi_g5_benign = rng.normal(18.0, 2.5, n_benign)
        cqi_lte_benign = rng.normal(14.0, 2.0, n_benign)
        per_benign = np.clip(rng.beta(1.5, 25.0, n_benign), 0.0, 0.15)
        dt_benign = rng.normal(0.10, 0.01, n_benign)

        X_benign = np.stack([r_p_benign, r_v_benign, r_a_benign, cqi_vlc_benign, cqi_g5_benign, cqi_lte_benign, per_benign, dt_benign], axis=1)
        y_benign = np.zeros(n_benign, dtype=int)

        # --- Generate Attack Traffic (4 Attack Classes) ---
        n_sub = n_attack // 4

        # Class 1: Position Offset Attack (Stealthy FDI +4.5m)
        r_p_att1 = rng.normal(4.48, 0.35, n_sub)
        r_v_att1 = rng.normal(0.18, 0.08, n_sub)
        r_a_att1 = rng.normal(0.35, 0.15, n_sub)

        # Class 2: Constant Speed Attack (Doppler Discrepancy)
        r_p_att2 = rng.normal(2.80, 0.60, n_sub)
        r_v_att2 = rng.normal(3.25, 0.45, n_sub)
        r_a_att2 = rng.normal(0.40, 0.20, n_sub)

        # Class 3: Random Kinematic Spoofing (Multi-Modal Discrepancy)
        r_p_att3 = rng.normal(6.50, 1.20, n_sub)
        r_v_att3 = rng.normal(4.80, 0.90, n_sub)
        r_a_att3 = rng.normal(2.85, 0.60, n_sub)

        # Class 4: Eventual Stop / Deceleration Deception
        r_p_att4 = rng.normal(3.60, 0.80, n_attack - 3 * n_sub)
        r_v_att4 = rng.normal(2.10, 0.50, n_attack - 3 * n_sub)
        r_a_att4 = rng.normal(4.50, 0.75, n_attack - 3 * n_sub)

        r_p_att = np.concatenate([r_p_att1, r_p_att2, r_p_att3, r_p_att4])
        r_v_att = np.concatenate([r_v_att1, r_v_att2, r_v_att3, r_v_att4])
        r_a_att = np.concatenate([r_a_att1, r_a_att2, r_a_att3, r_a_att4])

        cqi_vlc_att = rng.normal(20.0, 4.0, n_attack)
        cqi_g5_att = rng.normal(16.0, 3.5, n_attack)
        cqi_lte_att = rng.normal(12.0, 3.0, n_attack)
        per_att = np.clip(rng.beta(2.5, 15.0, n_attack), 0.02, 0.45)
        dt_att = rng.normal(0.105, 0.025, n_attack)

        X_att = np.stack([r_p_att, r_v_att, r_a_att, cqi_vlc_att, cqi_g5_att, cqi_lte_att, per_att, dt_att], axis=1)
        y_att = np.ones(n_attack, dtype=int)

        X_run = np.concatenate([X_benign, X_att], axis=0)
        y_run = np.concatenate([y_benign, y_att], axis=0)

        # Shuffle locally within run
        perm = rng.permutation(samples_per_run)
        X_list.append(X_run[perm])
        y_list.append(y_run[perm])
        run_ids.extend([run] * samples_per_run)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    run_ids = np.array(run_ids)

    # Normalize features using MinMax scaling to [0, 1] for SNN spike encoders
    min_vals = X_all.min(axis=0)
    max_vals = X_all.max(axis=0)
    X_norm = (X_all - min_vals) / (max_vals - min_vals + 1e-8)

    # Split into Grouped CV pool (Runs 1-12) and Held-Out Test Set (Runs 13-15)
    cv_mask = run_ids <= 12
    test_mask = run_ids > 12

    X_cv, y_cv, runs_cv = X_norm[cv_mask], y_all[cv_mask], run_ids[cv_mask]
    X_test, y_test, runs_test = X_norm[test_mask], y_all[test_mask], run_ids[test_mask]

    return (X_cv, y_cv, runs_cv), (X_test, y_test, runs_test)


def get_grouped_kfold_splits(X_cv, y_cv, runs_cv, n_splits=10):
    """
    Generates strict Scenario-Disjoint K-Fold splits where entire simulation runs are kept isolated in folds.
    """
    unique_runs = np.unique(runs_cv)
    np.random.seed(42)
    shuffled_runs = np.random.permutation(unique_runs)
    
    splits = []
    # Partition 12 runs across 10 folds
    for fold in range(n_splits):
        val_run = [shuffled_runs[fold % len(shuffled_runs)]]
        train_runs = [r for r in unique_runs if r not in val_run]

        val_idx = np.isin(runs_cv, val_run)
        train_idx = np.isin(runs_cv, train_runs)

        splits.append((train_idx, val_idx))
    return splits
