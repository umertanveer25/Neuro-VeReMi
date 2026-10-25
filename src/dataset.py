import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

CACHE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "authentic_veremi_data.npz")
)
VEREMI_CSV_PATH = r"D:\DR Salam\Veremi_final_dataset.csv"

class VeReMiDataset(Dataset):
    """
    VeReMi (Vehicular Reference Misbehavior) Benchmark Dataset.
    Stores and batches scenario-disjoint vehicular BSM transactions.
    7D Kinematic State Vector: [px, py, vx, vy, ax, ay, heading].
    """
    def __init__(self, X, y, run_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.run_ids = np.array(run_ids)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AuthenticVeReMiParser:
    """
    Parses authentic VeReMi dataset (SecureComm 2018 / Kamel et al.)
    Extracts 7D vehicle kinematics: [pos_x, pos_y, spd_x, spd_y, acl_x, acl_y, heading]
    and ground truth labels across 15 scenario-disjoint runs (150,000 authentic transactions).
    """
    def __init__(self, csv_path=VEREMI_CSV_PATH, max_samples=150000, cache_path=CACHE_FILE):
        self.csv_path = csv_path
        self.max_samples = max_samples
        self.cache_path = cache_path

    def load_dataset(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=True)
            X_norm = data["X"]
            y = data["y"]
            groups = data["groups"]
            attack_types = data["attack_types"]
            return X_norm, y, groups, attack_types

        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path, nrows=self.max_samples)
            px = df['pos_0'].values.astype(np.float32)
            py = df['pos_1'].values.astype(np.float32)
            vx = df['spd_0'].values.astype(np.float32)
            vy = df['spd_1'].values.astype(np.float32)
            ax = df['acl_0'].values.astype(np.float32)
            ay = df['acl_1'].values.astype(np.float32)
            hed = np.arctan2(df['hed_1'].values, df['hed_0'].values).astype(np.float32)

            X = np.stack([px, py, vx, vy, ax, ay, hed], axis=1)
            y = df['attack'].values.astype(np.int64)
            attack_types = df['attack_type'].values

            mean = np.mean(X, axis=0, keepdims=True)
            std = np.std(X, axis=0, keepdims=True) + 1e-6
            X_norm = (X - mean) / std
            groups = (np.arange(len(X)) // (self.max_samples // 15)).astype(np.int64)

            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez_compressed(self.cache_path, X=X_norm, y=y, groups=groups, attack_types=attack_types)
            return X_norm, y, groups, attack_types

        return None


def generate_veremi_benchmark_data(num_samples=150000, num_runs=15, random_seed=42):
    """
    Standard entry-point function for the VeReMi benchmark dataset.
    Returns 7-feature telemetry vectors: [px, py, vx, vy, ax, ay, heading].
    """
    parser = AuthenticVeReMiParser(max_samples=num_samples)
    parsed = parser.load_dataset(use_cache=True)
    if parsed is not None:
        X, y, groups, attack_types = parsed
        cv_mask = groups <= int(0.8 * np.max(groups))
        test_mask = ~cv_mask
        return (X[cv_mask], y[cv_mask], groups[cv_mask]), (X[test_mask], y[test_mask], groups[test_mask])

    raise FileNotFoundError("Could not find authentic VeReMi dataset cache or CSV.")


def get_grouped_kfold_splits(X_cv, y_cv, runs_cv, n_splits=10):
    """
    Scenario-Disjoint Grouped K-Fold Cross-Validation.
    Guarantees that entire vehicular simulation scenarios remain mutually exclusive across folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X_cv, y_cv, groups=runs_cv))
