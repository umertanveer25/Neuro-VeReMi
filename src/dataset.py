import os
import tarfile
import json
import glob
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

DEFAULT_VEREMI_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "V2X-BERT", "data", "veremi", "securecomm2018")
)
CACHE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "authentic_veremi_data.npz")
)

class AuthenticVeReMiParser:
    """
    Parses authentic VeReMi dataset tarballs (SecureComm 2018 / Kamel et al.)
    Extracts real vehicle kinematics: (pos_x, pos_y, spd_x, spd_y, acl_x, acl_y, heading)
    and true labels from GroundTruthJSONlog.json.
    """
    def __init__(self, veremi_dir=DEFAULT_VEREMI_DIR, max_samples=150000, cache_path=CACHE_FILE):
        self.veremi_dir = veremi_dir
        self.max_samples = max_samples
        self.cache_path = cache_path

    def load_dataset(self, use_cache=True):
        if use_cache and os.path.exists(self.cache_path):
            print(f"Loading cached authentic VeReMi data from: {self.cache_path}")
            data = np.load(self.cache_path)
            X_norm = data["X"]
            y = data["y"]
            groups = data["groups"]
            attack_types = data["attack_types"]
            print(f"Loaded {len(X_norm)} records. Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
            return X_norm, y, groups, attack_types

        tgz_files = sorted(glob.glob(os.path.join(self.veremi_dir, "*.tgz")))
        if not tgz_files:
            raise FileNotFoundError(f"No VeReMi .tgz files found in {self.veremi_dir}")

        features = []
        labels = []
        groups = []
        attack_types = []

        total_parsed = 0
        file_idx = 0

        print(f"Parsing {len(tgz_files)} authentic VeReMi archives...")
        for tgz_path in tgz_files:
            if total_parsed >= self.max_samples:
                break
            
            try:
                with tarfile.open(tgz_path, "r:gz") as tar:
                    for member in tar:
                        if member.name.endswith("GroundTruthJSONlog.json"):
                            f = tar.extractfile(member)
                            if f is None:
                                continue
                            
                            for line in f:
                                if total_parsed >= self.max_samples:
                                    break
                                try:
                                    record = json.loads(line.decode("utf-8"))
                                    pos = record.get("pos", [0.0, 0.0, 0.0])
                                    spd = record.get("spd", [0.0, 0.0, 0.0])
                                    acl = record.get("acl", [0.0, 0.0, 0.0])
                                    
                                    # Heading angle in 2D plane: atan2(vy, vx)
                                    heading = np.arctan2(spd[1], spd[0]) if len(spd) > 1 and (spd[0] != 0 or spd[1] != 0) else 0.0
                                    
                                    # State vector: [px, py, vx, vy, ax, ay, heading]
                                    feat = [
                                        float(pos[0]), float(pos[1]),
                                        float(spd[0]), float(spd[1]),
                                        float(acl[0]), float(acl[1]),
                                        float(heading)
                                    ]
                                    
                                    atk_type = int(record.get("attackerType", 0))
                                    is_malicious = 1 if atk_type > 0 else 0
                                    
                                    features.append(feat)
                                    labels.append(is_malicious)
                                    groups.append(file_idx)
                                    attack_types.append(atk_type)
                                    
                                    total_parsed += 1
                                except Exception:
                                    continue
                file_idx += 1
            except Exception as e:
                print(f"Warning: Could not process {tgz_path}: {e}")
                continue

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        groups = np.array(groups, dtype=np.int64)
        attack_types = np.array(attack_types, dtype=np.int64)

        # Standardize kinematic features
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True) + 1e-6
        X_norm = (X - mean) / std

        # Save cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            X=X_norm,
            y=y,
            groups=groups,
            attack_types=attack_types
        )
        print(f"Cached authentic dataset to {self.cache_path}")
        print(f"Successfully loaded {len(X)} authentic VeReMi records across {file_idx} scenarios.")
        print(f"Class distribution: Benign={np.sum(y==0)} ({np.mean(y==0)*100:.1f}%), Malicious={np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
        print(f"Attack Types Present: {np.unique(attack_types, return_counts=True)}")

        return X_norm, y, groups, attack_types

def get_grouped_kfold_splits(X, y, groups, n_splits=10):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups))

if __name__ == "__main__":
    parser = AuthenticVeReMiParser(max_samples=150000)
    X, y, groups, atk_types = parser.load_dataset(use_cache=False)
    print("Parsed X shape:", X.shape, "y shape:", y.shape)
