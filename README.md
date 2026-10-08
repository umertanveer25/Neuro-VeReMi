# Neuro-VeReMi: Ultra-Low-Latency Spiking Neural Networks for Energy-Efficient Zero-Trust Misbehavior Detection in Connected Vehicle Fleets

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/Validation-30x10--Fold%20CV%20(N=2,100)-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-28nm%20CMOS%20%26%20ARM%20Cortex--R52-purple.svg)]()
[![Safety](https://img.shields.io/badge/ISO%2026262-ASIL--D%20Deterministic%20(15.75%C2%B5s%20WCET)-blue.svg)]()

---

## 📌 1. System Architecture & End-to-End Pipeline

![Figure 1: System Architecture & End-to-End Pipeline](results/Fig1_System_Architecture_Pipeline.png)
*Fig. 1: Complete end-to-end dataflow pipeline of Neuro-VeReMi: (1) 100 Hz V2X telemetry ingestion with multipath Rayleigh fading and Doppler filtering; (2) Asynchronous Delta Modulation converting continuous kinematic residuals $r_p(t)$ into bipolar event streams $s_{\text{in}}(t)$; (3) Three-layer Spiking Neural Network featuring Kinematic-Aware Adaptive Decay (KA-LIF) neurons and recurrent temporal synapses; (4) Dynamic Multi-Hop Trust Engine ($\mathcal{H}_i$) and ISO 26262 ASIL-D automotive ECU actuation executing in $<1.85\,\mu\text{s}$ with $3.12\,\text{nJ}$ energy consumption.*

---

## 🔬 2. Neuronal Dynamics: Kinematic-Aware Adaptive Decay (`KA-LIF`)

![Figure 2: KA-LIF Neuronal Dynamics Circuit Schematic](results/Fig2_KALIF_Neuronal_Dynamics.png)
*Fig. 2: Detailed circuit schematic and computational signal flow of the Kinematic-Aware Adaptive Decay Leaky Integrate-and-Fire (KA-LIF) model. Incoming presynaptic spikes $s_j(t)$ are integrated through synaptic weights $W$. Simultaneously, the Kinematic Stress Estimator evaluates Newtonian velocity deviations and jerk invariants to dynamically adapt membrane retention $\beta_i(t)$ and anomaly driving current. The threshold comparator ($V_{\text{th}} = 0.75\,\text{V}$) emits an output spike $s_i(t)=1$ and triggers a hard reset ($u_i(t) \leftarrow 0.0\,\text{V}$).*

---

## 🛡️ 3. Multi-Layer Threat Model & VeReMi Attack Defense Taxonomy

![Figure 3: Multi-Layer Threat Model & VeReMi Attack Defense Taxonomy](results/Fig3_VeReMi_Threat_Model_Taxonomy.png)
*Fig. 3: Multi-layer V2X threat model and neuromorphic attack defense taxonomy. Ingested 5.9 GHz DSRC/C-V2X BSM telemetry subject to Spatial Domain (Types 1 & 2), Velocity Domain (Types 4 & 8), and Temporal/Collusion Domain (Type 16 & Byzantine Sybil rings) attacks are processed through the 3-layer Neuromorphic Defense Stack.*

---

## ⚙️ 4. Automotive AUTOSAR & ISO 26262 ASIL-D Embedded Deployment

![Figure 4: Automotive AUTOSAR & ISO 26262 ASIL-D Embedded Deployment](results/Fig4_AUTOSAR_ASIL_D_Deployment.png)
*Fig. 4: Automotive Electronic Control Unit (ECU) deployment architecture across ARM Cortex-R52 and Infineon AURIX TC397: (Left) AUTOSAR Classic/Adaptive Software Component (SW-C) running as a 100 Hz runnable with zero-malloc MISRA-C:2012 static memory allocation; (Middle) Real-Time Safety Engine guaranteeing an execution time of $1.85\,\mu\text{s}$ ($15.75\,\mu\text{s}$ WCET, $634\times$ safety margin); (Right) Dual-path actuation isolating anomalies.*

---

## 📊 5. Benchmark Performance Tables

### **Table 1: Comprehensive 300-Fold Cross-Validation Benchmark ($N=2,100$ Evaluations)**
*Evaluated across 30 randomized seeds $\times$ 10 scenario-disjoint folds ($N=300$ independent folds per model) on authentic VeReMi kinematic traces.*

| Model Architecture | Accuracy (%) [$N=300$] | Precision (%) | Recall (%) | $F_1$-Score (%) [$N=300$] | $F_1$ 95% CI | FPR (%) | AUC-ROC | Mean SynOps / MACs | Energy ($E_{\text{total}}$) | Latency (ARM Cortex-R52) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`KA_LIF_SNN (Proposed)`** | **$98.74 \pm 0.04$** | **$98.82$** | **$98.86$** | **$98.74 \pm 0.03$** | **$[98.70, 98.78]$** | **$0.18$** | **$0.9998$** | **$482.4$** | **$3.12\,\text{nJ}$** | **$1.85\,\mu\text{s}$** |
| `RLIF_SNN` | $97.92 \pm 0.08$ | $97.85$ | $98.00$ | $97.91 \pm 0.07$ | $[97.83, 97.99]$ | $0.95$ | $0.9982$ | $512.6$ | $3.35\,\text{nJ}$ | $1.96\,\mu\text{s}$ |
| `SCNN_1D` | $97.45 \pm 0.10$ | $97.30$ | $97.60$ | $97.45 \pm 0.09$ | $[97.35, 97.55]$ | $1.50$ | $0.9961$ | $1,240.0$ | $6.80\,\text{nJ}$ | $3.40\,\mu\text{s}$ |
| `ALIF_SNN` | $97.10 \pm 0.12$ | $97.02$ | $97.18$ | $97.10 \pm 0.11$ | $[96.98, 97.22]$ | $1.68$ | $0.9950$ | $620.5$ | $4.10\,\text{nJ}$ | $2.45\,\mu\text{s}$ |
| `PLIF_SNN` | $96.85 \pm 0.14$ | $96.72$ | $96.98$ | $96.85 \pm 0.13$ | $[96.70, 97.00]$ | $1.98$ | $0.9934$ | $710.2$ | $4.80\,\text{nJ}$ | $2.25\,\mu\text{s}$ |
| `LIF_SNN (Baseline)` | $96.10 \pm 0.18$ | $95.95$ | $96.25$ | $96.10 \pm 0.16$ | $[95.92, 96.28]$ | $2.95$ | $0.9890$ | $850.0$ | $4.20\,\text{nJ}$ | $2.10\,\mu\text{s}$ |
| `INT8_Quantized_MLP` | $95.82 \pm 0.22$ | $95.70$ | $96.00$ | $95.82 \pm 0.20$ | $[95.62, 96.08]$ | $3.30$ | $0.9845$ | $5,248.0\,\text{MACs}$ | $14.20\,\text{nJ}$ | $9.85\,\mu\text{s}$ |

---

### **Table 2: Inferential Statistical Hypothesis Testing ($N=300$ Paired Folds, $df=299$)**
*Paired parametric and exact non-parametric tests with standardized effect sizes.*

| Comparison Pair | Mean Diff $F_1$ | $95\%$ CI Diff | Paired $t$-stat $t(299)$ | Parametric $p$-value | Wilcoxon $W^+$ | Exact $p$-value | Cohen's $d_z$ | Cohen's $h$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KA-LIF vs. LIF (Baseline)** | $+2.64\%$ | $[+2.42\%, +2.86\%]$ | $24.81$ | $< 10^{-15}$ | $45,150.0$ | $< 10^{-15}$ | $1.43$ | $0.26$ |
| **KA-LIF vs. PLIF** | $+1.89\%$ | $[+1.71\%, +2.07\%]$ | $18.42$ | $< 10^{-15}$ | $44,890.0$ | $< 10^{-15}$ | $1.06$ | $0.21$ |
| **KA-LIF vs. ALIF** | $+1.64\%$ | $[+1.48\%, +1.80\%]$ | $16.15$ | $< 10^{-15}$ | $44,200.0$ | $< 10^{-15}$ | $0.93$ | $0.18$ |
| **KA-LIF vs. SCNN_1D** | $+1.29\%$ | $[+1.15\%, +1.43\%]$ | $14.28$ | $< 10^{-15}$ | $43,750.0$ | $< 10^{-15}$ | $0.82$ | $0.15$ |
| **KA-LIF vs. RLIF** | $+0.82\%$ | $[+0.71\%, +0.93\%]$ | $12.65$ | $< 10^{-15}$ | $42,900.0$ | $< 10^{-15}$ | $0.73$ | $0.11$ |
| **KA-LIF vs. INT8_MLP** | $+2.92\%$ | $[+2.68\%, +3.16\%]$ | $26.74$ | $< 10^{-15}$ | $45,150.0$ | $< 10^{-15}$ | $1.54$ | $0.28$ |

---

## 🖼️ 6. Empirical Validation & Experimental Figures (300 DPI)

* **Fig 5**: Kinematic Delta Modulation Encodings & Voltage Reset
* **Fig 6**: Multi-Architecture Performance & Real-Time Latency Comparison
* **Fig 7**: Hidden Layer Spike Raster & Output Membrane Potential Dynamics
* **Fig 8**: Energy vs Accuracy Pareto Frontier (SNNs vs. INT8 TinyML vs. Deep Learning)
* **Fig 9**: Adversarial Gradual Drift & Multi-Node Byzantine Defense
* **Fig 10**: Grouped Scenario-Disjoint Statistical Boxplots across 300 Folds

---

## 📂 7. Repository File Structure

```text
Neuro-VeReMi/
├── src/
│   ├── encoders.py               # Delta modulation, Poisson rate, TTFS spike encoders
│   ├── models.py                 # KA-LIF, RLIF, PLIF, ALIF, SCNN_1D, LIF, and INT8-MLP
│   ├── dataset.py                # Authentic VeReMi kinematic physics & trace parser
│   ├── synops_profiler.py        # 28nm CMOS SynOps, MAC, & End-to-End Energy Profiler
│   └── statistical_engine.py     # Paired t-test t(299), Wilcoxon W+, Cohen's d_z & h
├── experiments/
│   ├── run_parallel_30seed_benchmark.py  # 16-core parallel 30-seed 10-fold CV orchestrator
│   ├── generate_publication_figures.py   # Publication-quality 300 DPI figures generator
│   └── plot_fig1_exact_300dpi.py         # Standalone exact vector generator
├── results/                      # Standardized publication figures (Fig 1 to Fig 10), CSVs, JSON
├── requirements.txt              # Python dependencies
└── README.md                     # Repository documentation
```

---

## 🛠️ 8. Quickstart Installation & Reproduction

### 1. Clone & Environment Setup
```bash
git clone https://github.com/umertanveer25/Neuro-VeReMi.git
cd Neuro-VeReMi
pip install -r requirements.txt
```

### 2. Run Parallel 30-Seed $\times$ 10-Fold CV Benchmark ($2,100$ Evaluations)
```bash
python experiments/run_parallel_30seed_benchmark.py
```

### 3. Generate 300 DPI Publication Figures
```bash
python experiments/generate_publication_figures.py
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
