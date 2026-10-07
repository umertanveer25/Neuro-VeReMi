# REVIEWER 2 REPORT: INDEPENDENT EXPERT PEER REVIEW

**Manuscript Title:** *Neuro-VeReMi: Ultra-Low-Latency Spiking Neural Networks for Energy-Efficient Zero-Trust Misbehavior Detection in Connected Vehicle Fleets*  
**Review Type:** Comprehensive Technical & Methodological Independent Peer Review  
**Target Venue Standard:** IEEE Transactions on Intelligent Transportation Systems / IEEE Transactions on Vehicular Technology

---

## A. Overall Recommendation

**Recommendation: Major Revision**

**Justification:**  
The manuscript addresses a critical challenge in Connected and Autonomous Vehicle (CAV) security: reducing the energy consumption and inference latency of edge-based Zero-Trust misbehavior detection systems. The conceptual proposal of converting vehicular kinematic telemetry into sparse temporal spike trains via Spiking Neural Networks (SNNs) is promising and supported by an exhaustive 30-seed $\times$ 10-fold cross-validation framework ($N=300$ paired folds). 

However, acceptance cannot be granted in its present form due to several critical methodological and empirical concerns:
1. **Synthetic Data Discrepancy:** The experimental validation relies on parametrically synthesized Gaussian distributions generated via NumPy random seeds rather than parsing genuine simulation traces from the standardized VeReMi benchmark (Veins/SUMO/LuST).
2. **Analytical vs. Measured Hardware Claims:** Latency ($0.42\,\text{ms}$) and energy ($>90\%$ reduction) claims are presented with physical hardware certainty, despite being derived from first-order analytical multipliers ($0.9\,\text{pJ}$ Accumulate vs. $4.6\,\text{pJ}$ MAC) rather than physical silicon measurements.
3. **Absence of Modern Quantized Baselines:** Comparisons are made solely against unquantized FP32 dense ANNs, omitting standard automotive edge baselines such as INT8-quantized Multi-Layer Perceptrons (TinyML).

These issues are substantial but addressable through targeted revisions, empirical trace integration, and baseline expansion.

---

## B. Overall Assessment

The manuscript proposes **Neuro-VeReMi**, an event-driven framework applying temporal Spiking Neural Networks (LIF, PLIF, ALIF, 1D-SCNN, and Recurrent LIF) to vehicular misbehavior detection over Basic Safety Messages (BSMs). The authors develop three temporal spike encoding mechanisms (Delta Modulation, Poisson Rate, TTFS Latency) to translate continuous kinematic invariants (position, Doppler velocity, acceleration residuals, channel qualities) into binary spike sequences.

The work exhibits clear strengths in its statistical discipline: running 30 randomized splits of 10-fold cross-validation ($1,500$ independent model evaluations) with rigorous paired Student's $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$, and Cohen's $h$ effect sizes. The recurrence mechanism in R-LIF is demonstrated to significantly outperform feedforward spiking topologies. 

Nevertheless, the manuscript suffers from three major flaws: (i) an unverified synthetic feature generator standing in for the true VeReMi benchmark; (ii) overclaimed novelty regarding standard, pre-existing SNN neuron models (PLIF, ALIF, FastSigmoid surrogate gradients); and (iii) the omission of data preprocessing/encoding overheads from the total on-chip energy budget. A Major Revision is necessary to align the claims with rigorous empirical evidence.

---

## C. Strengths

1. **Rigorous Statistical Verification Protocol:** The execution of a 30-seed $\times$ 10-fold cross-validation framework ($N=300$ evaluation folds per architecture) with parametric ($t$-tests), non-parametric (exact Wilcoxon), and effect size metrics ($d_z, h$) sets a commendable benchmark for statistical reporting in vehicular security.
2. **Systematic Multi-Spiking Topology Benchmarking:** Comparing standard LIF, Parametric LIF (learnable $\beta$), Adaptive-Threshold LIF (ALIF), 1D Spatio-Temporal SCNN, and Recurrent LIF (R-LIF) provides valuable empirical insight into how different neuronal dynamics handle temporal vehicular momentum.
3. **Clean and Reproducible Codebase:** The codebase in `src/` is well-modularized, numerically stable, and provides explicit implementations of surrogate gradient backpropagation (FastSigmoid autograd functions) and analytical SynOps tracking.
4. **Relevant Practical Motivation:** Addressing the real-time ($<10\,\text{ms}$) and low-power constraints of automotive On-Board Units (OBUs) through event-driven neuromorphic accumulation is a pertinent direction for future CAV architectures.

---

## D. Major Concerns & Actionable Solutions

---

### Major Concern 1 — Parametric Synthetic Generator Used in Place of Official VeReMi Dataset
* **Location:** Section IV-A (*VeReMi Benchmark & Validation Protocol*) and `src/dataset.py` (lines 31–130).
* **Observation:** The manuscript claims evaluation on the "standardized VeReMi benchmark". However, inspection of `src/dataset.py` reveals that the dataset is generated using synthetic Gaussian/Beta random sampling:
  ```python
  r_p_benign = np.abs(rng.normal(0.12, 0.08, n_benign))
  r_p_att1 = rng.normal(4.48, 0.35, n_sub)
  per_benign = np.clip(rng.beta(1.5, 25.0, n_benign), 0.0, 0.15)
  ```
  No actual VeReMi JSON trace files (from the Veins/SUMO simulation environment) are ingested or parsed.
* **Technical Concern:** Synthetic parametric distributions lack the real-world artifacts of vehicular wireless channels, including multi-path fading, genuine non-stationary traffic burstiness, packet collisions, and realistic driver trajectory deviations. Evaluating on synthetic distributions risks reporting optimistic performance metrics that do not transfer to real vehicular networks.
* **Impact:** Undermines the validity of the empirical claims regarding VeReMi benchmark performance.
* **Required Clarification & Actionable Solution:**
  1. **Action:** Implement an authentic trace loader (`src/veremi_raw_loader.py`) that parses official VeReMi benchmark `.json` trace logs for key attack scenarios (e.g., Attack 1: Constant Position Offset, Attack 2: Random Offset, Attack 4: Eventual Stop, Attack 16: Delayed Messages).
  2. **Reframing:** Retain the current synthetic generator but explicitly re-label it in the text as a *Synthetic Kinematic Stress-Testing Protocol* designed to evaluate noise resilience across controlled SNR variations.
* **Severity:** **Critical**

---

### Major Concern 2 — Analytical Hardware Estimates Presented as Physical Silicon Facts
* **Location:** Abstract, Section I-A, Section V, and `src/synops_profiler.py`.
* **Observation:** The manuscript asserts that R-LIF achieves *"sub-millisecond ($0.42\,\text{ms}$) on-chip latency"* and *"$>90\%$ energy reduction"* without explicitly noting that these metrics are derived from theoretical multiplier models:
  $$\text{Energy} = \text{SynOps} \times 0.9\,\text{pJ}, \quad \text{Cycles} = \text{SynOps} \times 1.2 + 150$$
* **Technical Concern:** Analytical accounting based on the Horowitz (ISSCC 2014) 28nm standard cell table ignores memory access overheads (SRAM read/write energy), routing/interconnect energy on neuromorphic crossbars, spike packet serialization, and leakage power across idle cores.
* **Impact:** Misleads the reader into believing the architecture was verified on physical hardware (e.g., Intel Loihi 2 or SynSense DynapSE).
* **Required Clarification & Actionable Solution:**
  1. **Action:** Modify all text, abstracts, and table captions to explicitly state: *"Modeled Energy and Simulated Latency on an idealized 28nm Neuromorphic Baseline."*
  2. **Empirical Edge Validation:** Benchmark wall-clock execution time and memory footprint on an accessible physical embedded platform (e.g., Raspberry Pi 4, Jetson Nano, or ARM Cortex-M/R microcontroller) using ONNX Runtime or PyTorch C++ to provide real-world computational metrics.
* **Severity:** **High**

---

### Major Concern 3 — Missing Modern Quantized Integer (TinyML) Baselines
* **Location:** Section I-A (Introduction) and Section IV (Experimental Evaluation).
* **Observation:** The SNN is compared only against an unquantized FP32 dense ANN baseline. 
* **Technical Concern:** In automotive and embedded computing, dense floating-point ANNs are rarely deployed in FP32; standard practice utilizes 8-bit or 4-bit integer quantization (INT8/INT4 Post-Training Quantization or Quantization-Aware Training). INT8 MACs consume significantly less energy than FP32 MACs ($E_{\text{INT8-MAC}} \approx 0.2\,\text{pJ}$ on 28nm). Comparing an SNN against an unoptimized FP32 baseline creates an exaggerated efficiency delta.
* **Impact:** Incomplete baseline selection weakens the claim of architectural superiority.
* **Required Clarification & Actionable Solution:**
  1. **Action:** Add an **INT8 Quantized MLP (TinyML)** baseline to the comparative evaluation table.
  2. **Analysis:** Highlight that while INT8 ANNs execute dense matrix multiplications on every message cycle, SNNs benefit from **event-driven temporal sparsity** (zero-skipping), outperforming INT8 ANNs specifically during quiescent periods where spike rates drop below $20\%$.
* **Severity:** **High**

---

### Major Concern 4 — Omission of Temporal Spike Encoding Energy Overhead
* **Location:** Section III-B (*Temporal Spike Encoders*) and Section V (*SynOps Accounting*).
* **Observation:** The hardware energy accounting tracks only the synaptic operations (SynOps) within the hidden and output layers, ignoring the computational cost of the spike encoders (Delta Modulation, Poisson random number generation, TTFS latency sorting).
* **Technical Concern:** Continuous calculation of temporal derivatives ($x(t) - x(t-1)$) and threshold comparisons across $D=8$ features consumes ALU cycles and memory bandwidth. For shallow SNN architectures, the encoding overhead can constitute $15\text{--}35\%$ of total inference energy.
* **Impact:** Incomplete energy accounting underestimates end-to-end power consumption.
* **Required Clarification & Actionable Solution:**
  1. **Action:** Add the input encoding operations ($N_{\text{enc}} = D \times \text{Ops}_{\text{encode}}$) into the total energy equation:
     $$E_{\text{total}} = E_{\text{encoding}} + E_{\text{SynOps}} + E_{\text{leakage}}$$
  2. **Discussion:** Provide a brief breakdown demonstrating that Delta Modulation is computationally cheaper ($1$ subtraction + $1$ comparison) than stochastic Poisson encoding ($1$ PRNG call + $1$ float comparison).
* **Severity:** **Moderate**

---

### Major Concern 5 — Overclaiming Novelty of Standard SNN Mechanics
* **Location:** Section I-A (*Key Contributions*) and Section III (*SNN Dynamics*).
* **Observation:** The authors claim the introduction of PLIF, ALIF, and FastSigmoid surrogate gradients as novel components of their framework.
* **Technical Concern:** These components are verbatim implementations of established works: PLIF (Fang et al., ICCV 2021), ALIF (Bellec et al., Nature Comms, 2020), and FastSigmoid (Neftci et al., 2019).
* **Impact:** Overstating methodological novelty exposes the paper to immediate rejection by neuromorphic reviewers.
* **Required Clarification & Actionable Solution:**
  1. **Action:** Reframe the contributions to emphasize **domain application and empirical benchmarking**: *"We formulate the first comprehensive framework adapting temporal SNNs to vehicular Zero-Trust kinematics..."*
  2. **Architectural Grounding:** Introduce a domain-tailored mechanism, such as **Kinematic-Aware Adaptive Decay (KA-LIF)**, where the neuronal membrane decay factor $\beta$ adapts dynamically as a function of instantaneous acceleration magnitude:
     $$\beta(t) = \beta_0 \cdot \exp\left(-\lambda \cdot \frac{|a(t)|}{a_{\max}}\right)$$
* **Severity:** **Moderate**

---

## E. Minor Concerns

1. **Location:** Section II, Equation (1).  
   **Issue:** The trust history parameter $\mathcal{H}_i$ is listed in the input vector but lacks a formal mathematical update equation.  
   **Suggested Correction:** Define how $\mathcal{H}_i$ is updated (e.g., exponential moving average of past verification results).

2. **Location:** Section III-A, Equation (4).  
   **Issue:** The reset mechanism in the discrete LIF equation is written with subtractive reset ($- S_i[t-1] V_{\text{th}}$), but the PyTorch code uses hard reset ($U \cdot (1 - S)$).  
   **Suggested Correction:** Harmonize the text equation with the hard-reset implementation in `src/models.py`.

3. **Location:** Table I Caption.  
   **Issue:** Does not explicitly state whether the reported standard deviations are computed across all $N=300$ folds or across the 30 seed averages.  
   **Suggested Correction:** Clarify: *"Mean $\pm$ Standard Deviation across $N=300$ independent validation folds."*

4. **Location:** Section IV-B.  
   **Issue:** $p$-values are reported as "$p < 10^{-15}$".  
   **Suggested Correction:** Report exact degrees of freedom: *"paired $t(299) = 24.81, p < 10^{-15}$"*.

---

## F. Technical Verification

* **Mathematical Formulations:** Equations for LIF, ALIF, PLIF, and surrogate gradients are mathematically sound and correctly mapped to PyTorch autograd tensors.
* **Surrogate Gradient Backward Pass:** Verified in `src/models.py` line 23: $\frac{\partial \sigma}{\partial u} = \frac{1}{(1 + k |u - V_{\text{th}}|)^2}$. The FastSigmoid derivative is correct.
* **Grouped Scenario-Disjoint K-Fold Split:** Verified in `src/dataset.py` lines 131–149. The grouping logic prevents intra-scenario data leakage across folds.
* **Statistical Tests:** Verified in `src/statistical_engine.py`. Paired Student's $t$-test, Wilcoxon signed-rank test (exact), Cohen's $d_z$, and Cohen's $h$ are computed accurately.
* **SynOps Counter:** Verified in `src/models.py`. Counts active spikes multiplied by fan-out connections, which matches standard neuromorphic accounting.

---

## G. Results and Evidence Assessment

| Manuscript Claim | Evidence Provided | Evidence Adequate? | Technical Concern / Gap |
| :--- | :--- | :--- | :--- |
| **Claim 1: R-LIF achieves $97.15\%$ F1 on VeReMi.** | 300-fold cross-validation table. | **Partially** | Evaluated on synthetic Gaussian approximations, not raw VeReMi simulation traces. |
| **Claim 2: R-LIF outperforms feedforward SNNs ($p < 10^{-15}$).** | Paired $t$-tests and Wilcoxon tests with $d_z > 1.8$. | **Yes** | Statistically verified across 300 paired folds. |
| **Claim 3: $>90\%$ energy reduction vs. ANNs.** | Theoretical SynOps $\times$ $0.9\,\text{pJ}$ vs. MACs $\times$ $4.6\,\text{pJ}$. | **Partially** | Modeled analytical estimate; ignores encoding overhead and SRAM/interconnect energy. |
| **Claim 4: Sub-millisecond ($0.42\,\text{ms}$) edge latency.** | Modeled cycle equation ($cycles / 400\,\text{MHz}$). | **Partially** | Theoretical estimate; requires physical on-board profiling. |

---

## H. Novelty Assessment

1. **What is genuinely new?** The empirical adaptation and multi-topology comparison of recurrent/adaptive SNNs for vehicular kinematic telemetry anomaly detection.
2. **What appears incremental?** Applying standard SNN models (PLIF, ALIF) directly to tabular kinematic features.
3. **What is already established?** SNN neuron equations, FastSigmoid surrogate gradients, Delta modulation, and the Horowitz energy benchmarks.
4. **Is novelty adequately demonstrated?** **Partially.** Demonstrated as an applied engineering integration, but overstated as a fundamental architectural breakthrough.
5. **Final Novelty Rating:** **Moderate (3.5 / 10)**.

---

## I. Methodological Rigor Score

| Evaluation Category | Score (1–5) | Justification |
| :--- | :---: | :--- |
| **Problem Formulation** | 4 / 5 | Clear motivation addressing CAV edge latency and energy constraints. |
| **Literature Grounding** | 3 / 5 | Good coverage of classic VeReMi and SNNs; missing TinyML / quantized ANN literature. |
| **Methodological Rigor** | 4 / 5 | Sound PyTorch implementation of surrogate gradients and spiking dynamics. |
| **Dataset Quality** | 2 / 5 | Uses synthetic Gaussian feature distributions rather than genuine VeReMi trace logs. |
| **Experimental Design** | 5 / 5 | Exemplary 30-seed $\times$ 10-fold CV protocol ($N=300$ independent fold evaluations). |
| **Baseline Fairness** | 3 / 5 | Compares multiple SNNs, but lacks modern INT8 quantized edge baselines. |
| **Evaluation Metrics** | 4 / 5 | Comprehensive metrics (Acc, Prec, Rec, F1, FPR, AUC-ROC, SynOps, Energy). |
| **Statistical Analysis** | 5 / 5 | Exceptional parametric and non-parametric hypothesis testing with effect sizes. |
| **Reproducibility** | 4 / 5 | Codebase is clean, deterministic, and self-contained. |
| **Validity of Conclusions** | 3 / 5 | Hardware and benchmark claims are overstated relative to simulation reality. |

---

## J. Claim–Evidence Matrix

| Manuscript Claim | Supporting Evidence | Strength | Required Action |
| :--- | :--- | :--- | :--- |
| **"First Neuromorphic Zero-Trust V2X Architecture"** | SNN models applied to kinematic vectors. | Moderate | Reframe as an applied exploratory framework; moderate absolute "first" claim. |
| **"Sub-millisecond on-chip latency ($0.42\,\text{ms}$)"** | Analytical formula based on $400\,\text{MHz}$ clock. | Weak | Qualify as "Modeled Analytical Latency"; validate wall-clock time on embedded board. |
| **"$>90\%$ Energy Savings"** | Horowitz $0.9\,\text{pJ}$ SynOps calculation. | Moderate | Include encoding overhead and add INT8 Quantized baseline comparison. |
| **"R-LIF Superiority over Feedforward SNNs"** | $p < 10^{-15}$, Cohen's $d_z = 1.82\text{--}4.10$. | **Strong** | None. Well-supported by 300-fold experimental data. |

---

## K. Reproducibility Assessment

* **Reproducibility Level:** **High**
* **Assessment:** The provided scripts (`experiments/run_parallel_30seed_benchmark.py` and `src/`) are completely self-contained, fully deterministic, use explicit random seeds, and execute cleanly. Independent researchers can reproduce all reported numbers in under two minutes.

---

## L. Recommended Experiments

1. **Evaluation on Raw VeReMi Traces:**
   * *Hypothesis:* R-LIF maintains $>95\%$ F1-score when exposed to non-Gaussian noise and genuine simulation dropouts from Veins/SUMO.
   * *Metric:* Precision, Recall, F1-Score on official VeReMi Attack Types 1, 2, 4, 8, 16.
2. **Comparison with INT8 Quantized MLP (TinyML):**
   * *Hypothesis:* SNN achieves lower energy-delay product (EDP) than INT8-MLP under high temporal sparsity ($>80\%$).
   * *Metric:* SynOps vs. INT8 MACs, Energy (nJ), and F1-Score.
3. **Spike Encoding Overhead Ablation:**
   * *Hypothesis:* Delta Modulation incurs significantly lower computational overhead than Poisson stochastic coding.
   * *Metric:* Total end-to-end latency including encoder preprocessing.

---

## M. Required Revisions Before Acceptance

### Essential (Must Address for Re-evaluation)
1. Ingest and evaluate on actual raw VeReMi benchmark traces (or clearly designate synthetic data as a controlled parametric testbed).
2. Explicitly label all hardware energy and latency metrics as *analytical/modeled* estimates.
3. Implement an INT8 Quantized MLP baseline to provide a fair comparison against modern embedded edge techniques.
4. Correct novelty overclaims regarding pre-existing neuron models (PLIF, ALIF, FastSigmoid).

### Important (Significantly Improves Paper Quality)
1. Include the computational and energy cost of the temporal spike encoders in the overall energy accounting.
2. Formulate a domain-specific spiking enhancement (e.g., Kinematic-Aware Adaptive Decay LIF).
3. Harmonize mathematical equations (subtractive vs. hard reset) with the PyTorch codebase.

### Optional (Non-essential Enhancements)
1. Deploy and profile the model on an actual embedded edge board (e.g., Raspberry Pi 4 / STM32 NUCLEO).
2. Add a visualization comparing spike raster plots under benign kinematics vs. spoofing attacks.

---

## 29. Final Decision Summary

* **Recommendation:** **Major Revision**
* **Confidence in Recommendation:** **High**
* **Scientific Contribution:** **Moderate**
* **Technical Soundness:** **Moderate to High**
* **Novelty:** **Moderate**
* **Reproducibility:** **High**
* **Most Important Issue:** The experimental evaluation uses synthetic Gaussian feature distributions rather than genuine VeReMi simulation traces, and analytical hardware estimations are presented as measured physical facts.
* **Most Important Strength:** The 30-seed $\times$ 10-fold cross-validation protocol ($N=300$ folds per model) with paired parametric/non-parametric tests and effect size reporting is statistically exemplary.

### Top 3 Priority Actions for Authors:
1. **Integrate Genuine VeReMi Trace Data:** Ingest actual trace logs from the official VeReMi benchmark to substantiate performance claims on real vehicular network traffic.
2. **Qualify Hardware Claims & Add INT8 Baseline:** Clarify that energy/latency metrics are analytical 28nm models, and add an INT8 Quantized MLP baseline to prove SNN sparsity advantages over modern TinyML.
3. **Reframe Novelty Accurately:** Moderate claims of inventing standard SNN mechanics, framing the work as a pioneering domain-specific integration and empirical benchmark for neuromorphic V2X Zero-Trust security.
