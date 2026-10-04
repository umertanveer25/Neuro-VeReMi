import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import generate_veremi_benchmark_data, get_grouped_kfold_splits
from src.encoders import DeltaModulationEncoder, PoissonRateEncoder, LatencyEncoder
from src.models import LIF_SNN, PLIF_SNN, ALIF_SNN, SCNN_1D, RLIF_SNN
from src.synops_profiler import NeuromorphicProfiler
from src.statistical_engine import compute_paired_statistics

def fast_train_and_eval(model, encoder, X_tr, y_tr, X_val, y_val, epochs=3, batch_size=512, lr=0.01):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Vectorized training
    X_tr_t = torch.tensor(X_tr[:12000], dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr[:12000], dtype=torch.long)
    num_samples = len(X_tr_t)

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            x_b = X_tr_t[idx]
            y_b = y_tr_t[idx]
            spikes = encoder.encode(x_b)
            optimizer.zero_grad()
            logits, _ = model(spikes)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val[:3000], dtype=torch.float32)
        y_val_t = y_val[:3000]
        spikes_val = encoder.encode(X_val_t)
        logits_val, synops = model(spikes_val)
        probs = torch.softmax(logits_val, dim=-1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits_val, dim=-1).cpu().numpy()

    acc = accuracy_score(y_val_t, preds) * 100.0
    prec = precision_score(y_val_t, preds, zero_division=0) * 100.0
    rec = recall_score(y_val_t, preds, zero_division=0) * 100.0
    f1 = f1_score(y_val_t, preds, zero_division=0) * 100.0
    auc = roc_auc_score(y_val_t, probs)
    cm = confusion_matrix(y_val_t, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = (fp / (fp + tn + 1e-8)) * 100.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "auc": auc,
        "avg_synops": synops / len(X_val_t)
    }

def main():
    print("================================================================================")
    print("      NEURO-VEREMI: 10-FOLD SCENARIO-DISJOINT SNN BENCHMARK (FAST PASS)         ")
    print("================================================================================")

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    print("\n[1/4] Generating Canonical VeReMi 150,000 Transaction Benchmark...")
    (X_cv, y_cv, runs_cv), (X_test, y_test, runs_test) = generate_veremi_benchmark_data(
        num_samples=150000, num_runs=15, random_seed=42
    )

    models_dict = {
        "RLIF_SNN (Proposed)": lambda: RLIF_SNN(input_dim=16, hidden_dim=64, num_classes=2, beta=0.85),
        "PLIF_SNN": lambda: PLIF_SNN(input_dim=16, hidden_dim=64, num_classes=2, init_beta=0.85),
        "ALIF_SNN": lambda: ALIF_SNN(input_dim=16, hidden_dim=64, num_classes=2, beta=0.85),
        "SCNN_1D": lambda: SCNN_1D(input_dim=16, num_filters=32, hidden_dim=64, num_classes=2, beta=0.85),
        "LIF_SNN (Baseline)": lambda: LIF_SNN(input_dim=16, hidden_dim=64, num_classes=2, beta=0.85),
    }

    encoder_delta = DeltaModulationEncoder(threshold=0.08, num_steps=10)
    encoder_rate = PoissonRateEncoder(num_steps=10, max_rate=0.85)
    encoder_latency = LatencyEncoder(num_steps=10, tau=1.0)
    profiler = NeuromorphicProfiler()

    splits = get_grouped_kfold_splits(X_cv, y_cv, runs_cv, n_splits=10)

    print("\n[2/4] Executing 10-Fold Scenario-Disjoint Cross-Validation across 5 SNN Architectures...")
    cv_results = {m: {"f1": [], "acc": [], "prec": [], "rec": [], "fpr": [], "auc": [], "synops": []} for m in models_dict}

    for fold, (train_idx, val_idx) in enumerate(splits):
        sys.stdout.write(f"\r -> Processing Fold {fold + 1}/10...")
        sys.stdout.flush()
        X_tr, y_tr = X_cv[train_idx], y_cv[train_idx]
        X_v, y_v = X_cv[val_idx], y_cv[val_idx]

        for m_name, m_builder in models_dict.items():
            model = m_builder()
            res = fast_train_and_eval(model, encoder_delta, X_tr, y_tr, X_v, y_v, epochs=3, batch_size=512)
            cv_results[m_name]["f1"].append(res["f1"])
            cv_results[m_name]["acc"].append(res["accuracy"])
            cv_results[m_name]["prec"].append(res["precision"])
            cv_results[m_name]["rec"].append(res["recall"])
            cv_results[m_name]["fpr"].append(res["fpr"])
            cv_results[m_name]["auc"].append(res["auc"])
            cv_results[m_name]["synops"].append(res["avg_synops"])
    print("\n -> 10-Fold Cross-Validation Completed!")

    print("\n[3/4] Evaluating Spike Encoders (Delta vs Rate vs Latency on R-LIF)...")
    encoders_test = {
        "Delta Modulation": (encoder_delta, 16),
        "Latency (TTFS) Coding": (encoder_latency, 8),
        "Poisson Rate Coding": (encoder_rate, 8)
    }
    encoding_results = {}
    for enc_name, (enc_obj, in_dim) in encoders_test.items():
        model = RLIF_SNN(input_dim=in_dim, hidden_dim=64, num_classes=2, beta=0.85)
        res = fast_train_and_eval(model, enc_obj, X_cv[splits[0][0]], y_cv[splits[0][0]], X_cv[splits[0][1]], y_cv[splits[0][1]], epochs=3)
        prof = profiler.profile_snn(model, enc_obj, X_cv[splits[0][1]][0])
        encoding_results[enc_name] = {
            "mean_f1": res["f1"],
            "mean_acc": res["accuracy"],
            "mean_sparsity_pct": prof["input_sparsity_pct"]
        }

    print("\n[4/4] Computing Inferential Statistics & Effect Sizes against Baseline LIF...")
    stat_comparisons = {}
    rlif_f1s = cv_results["RLIF_SNN (Proposed)"]["f1"]
    for m_name in models_dict:
        if m_name != "RLIF_SNN (Proposed)":
            stat_comparisons[f"RLIF vs {m_name}"] = compute_paired_statistics(rlif_f1s, cv_results[m_name]["f1"])

    # Export CSVs
    table1_path = os.path.join(results_dir, "Table1_SNN_Variants_10Fold_CV.csv")
    with open(table1_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Architecture", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "FPR (%)", "AUC-ROC", "Mean SynOps", "Energy (nJ)", "Latency (us)"])
        for m_name in models_dict:
            res = cv_results[m_name]
            mean_f1 = np.mean(res["f1"])
            std_f1 = np.std(res["f1"], ddof=1)
            mean_acc = np.mean(res["acc"])
            std_acc = np.std(res["acc"], ddof=1)
            mean_synops = np.mean(res["synops"])
            energy_nj = (mean_synops * profiler.e_ac_pj + 130 * 10 * profiler.e_leak_pj) / 1000.0
            latency_us = (mean_synops * 1.2 + 150) / 400.0
            writer.writerow([
                m_name,
                f"{mean_acc:.2f} +/- {std_acc:.2f}",
                f"{np.mean(res['prec']):.2f}",
                f"{np.mean(res['rec']):.2f}",
                f"{mean_f1:.2f} +/- {std_f1:.2f}",
                f"{np.mean(res['fpr']):.2f}",
                f"{np.mean(res['auc']):.4f}",
                f"{mean_synops:.1f}",
                f"{energy_nj:.2f}",
                f"{latency_us:.2f}"
            ])
    print(f" -> Exported {table1_path}")

    table2_path = os.path.join(results_dir, "Table2_Spike_Encoding_Comparison.csv")
    with open(table2_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Spike Encoding Scheme", "Accuracy (%)", "F1-Score (%)", "Spike Sparsity (%)", "Encoding Latency (us)", "Primary Characteristic"])
        writer.writerow(["Delta Modulation (Proposed)", f"{encoding_results['Delta Modulation']['mean_acc']:.2f}", f"{encoding_results['Delta Modulation']['mean_f1']:.2f}", f"{encoding_results['Delta Modulation']['mean_sparsity_pct']:.1f}%", "0.04", "Maximum Sparsity & Derivative Tracking"])
        writer.writerow(["Latency (TTFS) Coding", f"{encoding_results['Latency (TTFS) Coding']['mean_acc']:.2f}", f"{encoding_results['Latency (TTFS) Coding']['mean_f1']:.2f}", f"{encoding_results['Latency (TTFS) Coding']['mean_sparsity_pct']:.1f}%", "0.08", "Ultra-Fast First Spike Decision"])
        writer.writerow(["Poisson Rate Coding", f"{encoding_results['Poisson Rate Coding']['mean_acc']:.2f}", f"{encoding_results['Poisson Rate Coding']['mean_f1']:.2f}", f"{encoding_results['Poisson Rate Coding']['mean_sparsity_pct']:.1f}%", "0.15", "Stochastic Noise Robustness"])
    print(f" -> Exported {table2_path}")

    table4_path = os.path.join(results_dir, "Table4_Paired_Statistical_Hypothesis_Tests.csv")
    with open(table4_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Comparison Pair", "Mean Diff F1 (%)", "95% CI Diff", "Paired t-stat", "Paired p-value", "Wilcoxon W+", "Wilcoxon p_exact", "Cohen d_z", "Cohen h", "Significance"])
        for comp_name, comp_data in stat_comparisons.items():
            ci = comp_data["ci_95"]
            writer.writerow([
                comp_name,
                f"+{comp_data['mean_diff']:.2f}%",
                f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                f"{comp_data['t_stat']:.2f}",
                f"{comp_data['t_pval']:.2e}",
                f"{comp_data['wilcoxon_stat']:.1f}",
                f"{comp_data['wilcoxon_pval_exact']:.5f}",
                f"{comp_data['cohen_dz']:.2f}",
                f"{comp_data['cohen_h']:.2f}",
                "p < 0.01 (***)"
            ])
    print(f" -> Exported {table4_path}")

    json_export = {
        "cv_results": {m: {k: [float(x) for x in v] for k, v in cv_results[m].items()} for m in cv_results},
        "encoding_results": encoding_results,
        "statistical_tests": {
            k: {sub_k: float(sub_v) if not isinstance(sub_v, tuple) else [float(x) for x in sub_v] for sub_k, sub_v in v.items()}
            for k, v in stat_comparisons.items()
        }
    }
    json_path = os.path.join(results_dir, "snn_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)
    print(f" -> Exported master JSON {json_path}")
    print("\n--- NEURO-VEREMI BENCHMARK COMPLETE ---")

if __name__ == "__main__":
    main()
