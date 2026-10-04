# Neuro-VeReMi: Ultra-Low-Latency Spiking Neural Networks for Energy-Efficient Zero-Trust Misbehavior Detection in Connected Vehicle Fleets

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/Validation-30x10--Fold%20CV-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Neuromorphic-SynOps%20%26%20Energy%20Modeled-purple.svg)]()

---

## 📌 Executive Abstract & Architecture

Connected and Autonomous Vehicles (CAVs) relying on Cooperative Awareness Messages (CAMs) over Basic Safety Messages (BSMs) are acutely vulnerable to cyber-physical spoofing, position alteration, and coordinated denial-of-service attacks. Traditional Deep Learning (DL) approaches require power-hungry floating-point multiply-accumulate (MAC) units and introduce latency bottlenecks unacceptable for real-time vehicular safety critical envelopes ($<10\,\text{ms}$).

**Neuro-VeReMi** introduces a novel bio-inspired, neuromorphic Zero-Trust misbehavior detection architecture leveraging **Spiking Neural Networks (SNNs)**:
* **Asynchronous Event-Driven Computing**: Transforms continuous vehicular telemetry into sparse, temporal binary spike trains via **Delta Modulation**, **Poisson Rate**, and **Time-to-First-Spike (TTFS)** latency encoders.
* **Recurrent & Adaptive SNN Architectures**: Evaluates Recurrent Leaky Integrate-and-Fire (`RLIF_SNN`), Parametric LIF (`PLIF_SNN` with learnable $\beta$), Adaptive Threshold LIF (`ALIF_SNN`), and 1D Spiking Convolutional Networks (`SCNN_1D`).
* **Ultra-Low Energy SynOps Profiling**: Replaces standard dense float MACs ($4.6\,\text{pJ}$) with event-driven synaptic additions ($E_{\text{AC}} = 0.9\,\text{pJ}$ on $28\,\text{nm}$ neuromorphic silicon), yielding **$>90\%$ energy reduction** and sub-millisecond on-chip inference.
* **Exhaustive Empirical Validation**: Validated across **30 randomized scenario-disjoint splits of 10-fold cross-validation** ($1,500$ independent model evaluations), paired Student's $t$-tests, exact Wilcoxon signed-rank tests, Cohen's $d_z$, and Cohen's $h$ effect sizes.

---

## 🚀 Key Results Preview

| Model Architecture | Precision (%) | Recall (%) | F1-Score (%) | SynOps / Inf | Energy ($\mu\text{J}$) | Inf. Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RLIF-SNN (Ours)** | **$97.42 \pm 0.31$** | **$96.88 \pm 0.35$** | **$97.15 \pm 0.28$** | **$18,420$** | **$0.0166\,\mu\text{J}$** | **$0.42\,\text{ms}$** |
| **ALIF-SNN** | $96.15 \pm 0.42$ | $95.60 \pm 0.48$ | $95.87 \pm 0.39$ | $19,850$ | $0.0179\,\mu\text{J}$ | $0.48\,\text{ms}$ |
| **PLIF-SNN** | $95.80 \pm 0.44$ | $95.12 \pm 0.51$ | $95.46 \pm 0.41$ | $21,100$ | $0.0190\,\mu\text{J}$ | $0.51\,\text{ms}$ |
| **1D-SCNN** | $94.90 \pm 0.52$ | $94.25 \pm 0.58$ | $94.57 \pm 0.49$ | $28,400$ | $0.0256\,\mu\text{J}$ | $0.68\,\text{ms}$ |
| **Standard LIF SNN** | $92.30 \pm 0.65$ | $91.80 \pm 0.70$ | $92.05 \pm 0.61$ | $16,900$ | $0.0152\,\mu\text{J}$ | $0.39\,\text{ms}$ |

---

## 📂 Repository Structure

```tree
Neuro-VeReMi/
├── src/
│   ├── encoders.py               # Delta modulation, Poisson rate, TTFS spike encoders
│   ├── models.py                 # LIF, PLIF, ALIF, 1D-SCNN, and Recurrent RLIF SNNs
│   ├── dataset.py                # Scenario-disjoint VeReMi dataset generator & loader
│   ├── synops_profiler.py        # Synaptic Operations (SynOps), Sparsity, & Energy Profiler
│   └── statistical_engine.py     # Paired t-tests, Wilcoxon signed-rank, Cohen's d_z & h
├── experiments/
│   ├── run_parallel_30seed_benchmark.py  # 16-core parallel 30-seed 10-fold CV orchestrator
│   └── generate_publication_figures.py   # Publication-quality 300 DPI figures generator
├── results/                      # Output CSV tables, JSON results, and figures
├── paper/                        # LaTeX manuscript, IEEEtran template, bibliography
├── requirements.txt              # Environment dependencies
└── README.md                     # Comprehensive documentation
```

---

## 🛠️ Quickstart Installation & Reproduction

### 1. Clone & Environment Setup
```bash
git clone https://github.com/umertanveer25/Neuro-VeReMi.git
cd Neuro-VeReMi
pip install -r requirements.txt
```

### 2. Run Parallel 30-Seed 10-Fold CV Benchmark (1,500 Evaluations)
```bash
python experiments/run_parallel_30seed_benchmark.py
```

### 3. Generate 300 DPI Publication Figures
```bash
python experiments/generate_publication_figures.py
```

---

## 📜 Citation

If you use Neuro-VeReMi in your research, please cite:
```bibtex
@article{tanveer2026neuroveremi,
  title={Neuro-VeReMi: Ultra-Low-Latency Spiking Neural Networks for Energy-Efficient Zero-Trust Misbehavior Detection in Connected Vehicle Fleets},
  author={Tanveer, Muhammad Umer and et al.},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
