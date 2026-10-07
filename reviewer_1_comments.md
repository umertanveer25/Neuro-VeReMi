# REVIEWER 1 COMMENTS: NOVELTY, CONTRIBUTION ASSESSMENT & ACTIONABLE SOLUTIONS

**Manuscript Title:** *Neuro-VeReMi: Ultra-Low-Latency Spiking Neural Networks for Energy-Efficient Zero-Trust Misbehavior Detection in Connected Vehicle Fleets*  
**Evaluation Scope:** Methodological Originality, Scientific Contribution, Prior-Art Differentiation, Empirical Validity, Claim Justification, and Actionable Solutions.

---

## 1. Independent Novelty Verdict

The manuscript explores an **application and integration-level contribution**: adapting established Spiking Neural Network (SNN) architectures (LIF, PLIF, ALIF, 1D-SCNN, and R-LIF) and standard event-encoding schemes (Delta Modulation, Poisson Rate, TTFS Latency) to kinematic telemetry classification for vehicular misbehavior detection (Zero-Trust V2X). 

From a purely methodological and theoretical perspective, **the manuscript introduces no fundamentally new neuron models, no new learning algorithms, and no new theoretical energy formulations**. The neuron dynamics (LIF, PLIF with learnable $\beta$, ALIF with threshold adaptation, R-LIF with recurrent synapses) and the FastSigmoid surrogate gradient are direct implementations of existing neuromorphic literature (Fang et al., 2021; Bellec et al., 2020; Neftci et al., 2019; Zenke & Vogels, 2021). Furthermore, the energy and hardware efficiency claims are derived via standard analytical SynOps accounting (Horowitz, 2014 energy benchmarks) rather than empirical measurements on physical neuromorphic silicon (such as Intel Loihi 2 or SynSense DynapSE). 

However, within the vehicular networking and intelligent transportation systems (ITS) domain, the work delivers a **solid engineering and empirical integration**. It demonstrates that recurrent spiking dynamics (R-LIF) can classify kinematic anomaly signatures while drastically reducing theoretical computational operations compared to dense floating-point ANNs. The statistical protocol (30 randomized seeds $\times$ 10-fold cross-validation, $N=300$ folds per model, paired parametric and non-parametric tests with effect sizes) is methodologically disciplined, though its empirical foundation is constrained by the use of synthetic kinematic feature distributions rather than raw vehicular network traces.

---

## 2. What Is Actually New?

1. **Application of Temporal SNNs to V2X Kinematic Anomaly Detection:** Formulating vehicular BSM telemetry (kinematic invariants, channel qualities, packet delivery statistics) as temporal spike trains for edge-oriented Zero-Trust verification.
2. **Comparative Encoder-Architecture Benchmarking in V2X:** A systematic comparative study evaluating how 3 temporal encoders (Delta Modulation, Poisson, TTFS) interface with 5 spiking topologies (LIF, PLIF, ALIF, SCNN, R-LIF) on vehicular kinematic vectors.
3. **High-Repetition Statistical Protocol in Neuromorphic V2X:** Executing a large-scale evaluation grid ($1,500$ model evaluations across 300 folds per architecture) with rigorous paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$, and Cohen's $h$ effect sizes.

---

## 3. What Is Not New?

1. **Neuron Dynamics and Architectures:**
   * *LIF / FastSigmoid Surrogate Gradient:* Well-established (Neftci et al., 2019; Zenke & Vogels, 2021).
   * *Parametric LIF (PLIF):* Proposed and formalized by Fang et al. (ICCV 2021).
   * *Adaptive-Threshold LIF (ALIF):* Introduced by Bellec et al. (Nature Comms, 2020) and Yin et al. (Nature Mach. Intell., 2021).
   * *Recurrent LIF (R-LIF):* Standard recurrent spiking formulation.
2. **Temporal Spike Encoders:**
   * *Delta Modulation:* Standard signal processing and event-based sensing encoding dating back decades.
   * *Poisson Rate Coding & TTFS Latency Coding:* Canonical neuromorphic coding paradigms widely documented across neuromorphic textbooks and frameworks (e.g., snnTorch, SpikingJelly).
3. **Neuromorphic SynOps and Energy Modeling:**
   * The equation $\text{SynOps} = \sum T \cdot N_{l-1} \cdot N_l \cdot \zeta_{l-1}$ and the energy constants ($E_{\text{AC}} = 0.9\,\text{pJ}$, $E_{\text{MAC}} = 4.6\,\text{pJ}$ on 28nm/45nm CMOS) are standard analytical accounting models taken directly from Horowitz (ISSCC 2014) and standard SNN literature.
4. **Vehicular Misbehavior Detection (V2X Zero-Trust):**
   * Detecting position, velocity, and deceleration tampering on BSM/CAM streams using the VeReMi formulation was established by van der Heijden et al. (IEEE T-ITS, 2018) and studied in dozens of subsequent machine learning publications.

---

## 4. Novelty Decomposition

| Pipeline Component | Status | Source / Evidence | Novelty Level | Role in Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **1. Kinematic Invariant State Vector** $\mathbf{x}(t)$ | Existing | van der Heijden et al. (2018); standard VeReMi features ($r_p, r_v, r_a, \text{CQI}, \text{PER}, \Delta t$) | None (Level 0) | Provides foundational input features. |
| **2. Temporal Spike Encoders (Delta/Poisson/TTFS)** | Existing | Standard neuromorphic encoding functions; standard threshold delta modulation | None (Level 0) | Transforms continuous inputs into binary event tensors. |
| **3. Spiking Neuron Dynamics (LIF, PLIF, ALIF, R-LIF)** | Existing | Fang et al. (2021); Bellec et al. (2020); standard PyTorch surrogate gradient implementations | None (Level 0) | Core classification engine; R-LIF captures sequential dependencies. |
| **4. FastSigmoid Surrogate Gradient Optimization** | Existing | Neftci et al. (2019); Zenke & Vogels (2021) | None (Level 0) | Enables standard backpropagation through non-differentiable threshold. |
| **5. SynOps / Energy Estimation Model** | Existing | Horowitz (2014) $0.9\,\text{pJ}$ vs $4.6\,\text{pJ}$ analytical model | None (Level 0) | Quantifies theoretical energy advantage of sparse binary events. |
| **6. Overall Integration & V2X SNN Benchmarking** | Modified / Integrated | Manuscript integration | **Level 1 (Application Novelty)** | Demonstrates feasibility and efficiency of SNNs for vehicular security. |

---

## 5. Closest Prior Work Comparison

| Prior Work | Approach / Methodology | Current Manuscript | Key Difference | Technical Meaningfulness |
| :--- | :--- | :--- | :--- | :--- |
| **van der Heijden et al. (2018)** *VeReMi Benchmark* | Introduced VeReMi dataset; evaluated rule-based and classic ML detectors. | Applies event-driven SNNs to kinematic features. | Replaces classic ML/rules with temporal spiking neural networks. | **Moderate** (Applies neuromorphic paradigm to vehicular benchmark). |
| **Alladi et al. (2021)** *Deep Learning for Secure VANET* | Used dense ANNs (DNN, CNN, LSTM) with FP32 MAC operations. | Uses sparse binary Spiking Neural Networks (R-LIF). | Eliminates dense FP32 MACs in favor of sparse Accumulations ($E_{\text{AC}}$). | **Moderate** (Shifts compute paradigm from dense ANN to sparse SNN). |
| **Yin et al. (2021) / Bellec et al. (2020)** *Recurrent & Adaptive SNNs* | Formulated ALIF and recurrent SNNs for temporal sequences (audio, gestures). | Applies recurrent/adaptive SNNs to tabular vehicular telemetry sequences. | Domain transfer from neuromorphic audio/vision to vehicular kinematics. | **Low to Moderate** (Straightforward domain transfer of known architectures). |
| **Fang et al. (2021)** *Parametric LIF (PLIF)* | Proposed learnable membrane decay $\beta$ via backpropagation for vision/audio. | Evaluates PLIF on vehicular telemetry vectors. | Direct application of the PLIF formulation without architectural alteration. | **Low** (Direct application of existing model). |

---

## 6. Authors' Claims vs. Independent Assessment

| Aspect | Authors' Stated Claim | Independent Technical Assessment | Discrepancy / Alignment |
| :--- | :--- | :--- | :--- |
| **Primary Novelty** | "Pioneering bio-inspired neuromorphic Zero-Trust misbehavior detection architecture." | Application of existing SNN algorithms to a domain previously dominated by ANNs/ML. | **Overstated**: Represents application/integration novelty, not a new architectural paradigm. |
| **Neuron & Algorithm Novelty** | Introduces novel Recurrent LIF, PLIF, and ALIF models for V2X. | Uses textbook implementations of standard SNN neurons and FastSigmoid surrogate gradients. | **Overstated**: All neuron equations and backpropagation mechanics are verbatim from prior literature. |
| **Empirical Scope** | "Exhaustive evaluation on standardized VeReMi benchmark." | Uses a synthetic mathematical feature generator with Gaussian/Beta distributions rather than raw VeReMi simulation traces. | **Substantially Overstated**: The data is synthetic feature approximations, not real VeReMi trace files. |
| **Hardware / Energy Claims** | "Sub-millisecond on-chip latency ($0.42\,\text{ms}$) and $>90\%$ energy reduction." | Theoretical estimates computed using analytical multiplication ($cycles = synops \times 1.2 + 150$; $E = SynOps \times 0.9\,\text{pJ}$). | **Overstated**: Modeled/simulated estimates presented with hardware-level certainty without physical chip deployment. |
| **Statistical Rigor** | Paired 30-seed $\times$ 10-fold CV ($N=300$) with effect sizes ($d_z, h$). | Protocol is rigorously executed in code and statistically sound. | **Accurate & Fully Aligned**. |

---

## 7. Contribution Assessment

### Contribution 1: Domain Application (SNNs for Vehicular Zero-Trust)
* **Classification:** Application / Integration Contribution.
* **Assessment:** Demonstrating that spiking networks can process vehicular kinematic anomalies with high accuracy ($F1 \approx 97.15\%$) is a constructive application. It provides proof-of-concept for edge-deployable neuromorphic automotive security.

### Contribution 2: Multi-Architecture Comparative Study (LIF, PLIF, ALIF, SCNN, R-LIF)
* **Classification:** Empirical Contribution.
* **Assessment:** Comparing multiple SNN variants shows that recurrent connections (R-LIF) provide significant performance gains ($p < 10^{-15}, d_z = 1.82\text{--}4.10$) over feedforward LIF and adaptive variants when temporal sequence momentum is involved.

### Contribution 3: Hardware SynOps and Energy Modeling
* **Classification:** Engineering / Analytical Contribution.
* **Assessment:** Useful for first-order estimation of on-chip energy dissipation, but lacks physical silicon validation (e.g., measurements on Loihi 2, BrainScaleS, or FPGA neuromorphic emulators).

---

## 8. Novelty and Contribution Scores

* **Overall Novelty:** **3.5 / 10** *(Limited / Incremental Application Novelty)*  
  *Justification:* The core components (neuron models, surrogate gradients, encoders, energy equations) are unmodified existing techniques from prior literature.
* **Scientific Contribution:** **4.0 / 10** *(Moderate Empirical Impact within V2X Domain)*  
  *Justification:* Provides empirical evidence of SNN suitability for V2X telemetry, but does not advance neuromorphic theory or machine learning methodology.
* **Practical / Engineering Contribution:** **6.5 / 10** *(Solid Proof-of-Concept for Embedded Automotive OBUs)*  
  *Justification:* Highlights a viable engineering pathway to reduce floating-point power overheads in vehicular edge units.
* **Theoretical Contribution:** **1.0 / 10** *(Minimal / None)*  
  *Justification:* No new theorems, proofs, or mathematical formulations beyond existing literature.

---

## 9. Major Novelty Concerns

1. **Synthetic Data vs. True VeReMi Traces:** In `src/dataset.py`, the function `generate_veremi_benchmark_data` generates Gaussian-distributed approximations of 8 invariant features (`rng.normal(...)`, `rng.beta(...)`). Claiming results on the "standardized VeReMi benchmark" when evaluating on synthetic Gaussian distributions creates an ungrounded performance claim.
2. **Absence of Real Neuromorphic Hardware Deployment:** All energy and latency numbers are derived from theoretical multipliers ($0.9\,\text{pJ}$ per Accumulate, $400\,\text{MHz}$ clock cycle formula) rather than actual deployment on neuromorphic hardware (e.g., Intel Loihi 2, SynSense Spear/DynapSE).
3. **No SNN-Specific Algorithmic Innovation:** The manuscript does not introduce a new surrogate loss, a novel spike-timing-dependent plasticity (STDP) rule, or a V2X-specific neuron dynamic. It is a direct pipeline assembly of existing techniques.

---

## 10. Evidence Required to Establish Higher Novelty

To elevate this work to a higher tier of novelty and contribution:
1. **Validation on Raw Real-World / Full VeReMi Datasets:** Ingest the actual JSON/FCD output logs from Veins/SUMO vehicular simulation runs rather than synthetic parametric distributions.
2. **Physical Neuromorphic Deployment:** Deploy the compiled spiking network onto an actual neuromorphic chip (e.g., Loihi 2 / DynapSE) or an FPGA neuromorphic core (e.g., AMD Xilinx Zynq) and measure real power dissipation using an oscilloscope/power profiler.
3. **Ablation on Encoding Dynamics:** Provide a dedicated empirical ablation proving whether temporal spike timing (e.g., TTFS phase) conveys information that standard quantized fixed-point ANNs (e.g., INT8/INT4 quantized MLPs) cannot capture at similar energy budgets.

---

## 11. Overclaiming Assessment

1. **"First Neuromorphic Architecture for V2X Misbehavior Detection":** While possibly the first paper specifically titling this combination on VeReMi features, SNNs have been applied broadly to network intrusion detection (NIDS) and IoT anomaly detection. The claim of architectural pioneering should be moderated to an application study.
2. **"Sub-millisecond on-chip latency ($0.42\,\text{ms}$)":** This must be explicitly qualified as *modeled/simulated latency on an idealized ARM/neuromorphic hardware model*, not measured on physical silicon.
3. **"Standardized VeReMi Benchmark Validation":** The paper must clearly disclose that it tests on a synthesized invariant feature representation inspired by VeReMi distributions, rather than raw trace logs.

---

## 12. Final Independent Verdict

### **Verdict: Primarily an Application/Integration Contribution (Novel but Incremental Application)**

### Summary of Independent Questions:
* **Q1. Is the manuscript genuinely novel?** **Partially.** Novel as a specific domain application and comparative evaluation in vehicular security, but lacks methodological or theoretical novelty.
* **Q2. Single most novel element:** The systematic empirical comparison of recurrent and adaptive spiking neural networks against standard LIF models for multi-variate vehicular kinematic anomaly detection.
* **Q3. Strongest contribution:** The rigorous 30-seed $\times$ 10-fold cross-validation statistical framework ($N=300$ folds, $p < 10^{-15}$, large effect sizes $d_z > 1.8$) demonstrating the consistency of R-LIF over simpler spiking baselines.
* **Q4. Weakest contribution claim:** The claim of "sub-millisecond on-chip latency and $>90\%$ energy reduction" presented as an established physical fact when it is an analytical first-order simulation.
* **Q5. Is the claimed research gap genuine?** **Partially.** The energy and latency constraints of edge CAVs are real, but comparisons against modern quantized integer ANNs (e.g., 4-bit/8-bit integer quantized MLPs) were omitted.
* **Q6. Is the contribution scientifically meaningful?** **Yes, within applied intelligent transportation systems**, as an exploratory benchmark for event-driven vehicular safety architectures.
* **Q7. Is the novelty overstated?** **Yes.** The manuscript presents standard SNN models and analytical energy accounting as proprietary architectural breakthroughs.
* **Q8. Is the contribution sufficiently demonstrated?** **Partially.** Statistically rigorous across folds, but limited by synthetic feature generation and lack of physical hardware measurements.
* **Q9. Does the manuscript provide evidence of advancement over the closest prior work?** **Yes, compared to non-spiking or standard feedforward LIF baselines**, demonstrating clear accuracy and sparsity benefits for recurrent spiking topologies on temporal telemetry sequences.

---

# 🛠️ CRITICISM & ACTIONABLE SOLUTIONS (WAY FORWARD)

---

### 1. Dataset Authenticity: Synthetic Generators vs. Real VeReMi Traces
* **Criticism:** `src/dataset.py` synthesizes features using Gaussian and Beta distributions (`rng.normal`, `rng.beta`) rather than parsing raw VeReMi simulation JSON/FCD traces. Reviewers will flag this as a critical methodology discrepancy.
* **Actionable Solution & Way Forward:**
  1. **Two-Tier Validation Strategy:**
     * **Tier 1 (Real Traces):** Integrate a loader for raw VeReMi benchmark JSON/CSV logs (e.g., Attack Types 1, 2, 4, 8, 16 from the official 2018 benchmark) to demonstrate performance on raw vehicular network traces.
     * **Tier 2 (Controlled Synthetic Protocol):** Keep the 30-seed $\times$ 10-fold CV generator, but explicitly designate it in the manuscript as a *Parametric Kinematic Invariant Benchmark* for stress-testing signal-to-noise degradation and edge corner-cases.
  2. **Code Implementation:** Add a clean parser module `src/veremi_raw_loader.py` that reads genuine `.json` logs containing `sender`, `pos`, `pos_noise`, `spd`, `spd_noise` from the official dataset repository.

---

### 2. Novelty Framing: Overclaiming Standard SNN Mechanics
* **Criticism:** Describing PLIF, ALIF, and FastSigmoid surrogate gradients as newly invented proprietary mechanics triggers immediate reviewer pushback, as these were established by Fang et al. (2021), Bellec et al. (2020), and Neftci et al. (2019).
* **Actionable Solution & Way Forward:**
  1. **Reframe the Contribution Ethos:** 
     * Shift framing from *"We invented a novel spiking architecture"* to *"We present the first domain-grounded neuromorphic framework that reformulates vehicular Zero-Trust kinematics into sparse temporal event streams."*
  2. **Introduce a Domain-Specific Spiking Innovation (Kinematic-Decay LIF):**
     * Instead of a static or purely learned $\beta$, formulate a **Kinematic-Aware Adaptive Decay (KA-LIF)** where the membrane decay constant $\beta(t)$ dynamically modulates with vehicular acceleration jitter:
       $$\beta(t) = \beta_0 \cdot \exp\left(-\lambda \cdot \frac{|\Delta v(t)|}{v_{\max}}\right)$$
     * This introduces a genuine, domain-specific methodological novelty linking vehicular physics directly to neuronal membrane dynamics.

---

### 3. Hardware Energy & Latency: Analytical Proxies vs. Silicon Reality
* **Criticism:** Stating "$0.42\,\text{ms}$ on-chip latency and $>90\%$ energy savings" as established hardware facts when they are derived from analytical multipliers ($cycles = synops \times 1.2 + 150$) will be rejected by hardware-savvy reviewers.
* **Actionable Solution & Way Forward:**
  1. **Adopt Standard Neuromorphic Terminology:**
     * State: *"Modeled Neuromorphic Silicon Accounting based on 28nm CMOS energy benchmarks (Horowitz, 2014) and cycle-accurate ARM Cortex-R52 execution models."*
  2. **Add Practical Edge Profiling (PyTorch Mobile / ONNX / TinyML):**
     * Measure actual wall-clock inference time and RAM consumption on an accessible embedded hardware board (e.g., Raspberry Pi 4 / Nvidia Jetson Nano / STM32 NUCLEO) to provide true empirical edge figures alongside theoretical SynOps.

---

### 4. Baseline Fairness: Floating-Point ANN Strawman vs. Quantized TinyML
* **Criticism:** Comparing an ultra-sparse SNN only against unquantized FP32 dense ANNs is often viewed as an unfair comparison, since production automotive OBUs frequently use INT8/INT4 quantized models.
* **Actionable Solution & Way Forward:**
  1. **Include Post-Training Quantized (INT8) Baseline:**
     * Implement an INT8 Quantized Multi-Layer Perceptron (INT8-MLP) baseline in `src/models.py`.
  2. **Highlight the Sparse-Event Advantage:**
     * Show that while INT8 ANNs perform 8-bit integer multiplications on every clock cycle, SNNs perform **zero-skipping additions only on active spike events**, achieving lower Total Energy-Delay Product (EDP) when input sparsity exceeds $80\%$.

---

### 5. Physical Grounding of Temporal Encoders
* **Criticism:** The multi-encoder comparison (Delta vs. Poisson vs. Latency) was presented purely as a machine learning hyperparameter choice without physical motivation.
* **Actionable Solution & Way Forward:**
  1. **Physically Ground Delta Modulation:**
     * Provide the differential physics rationale: vehicular telemetry sensors measure Newtonian dynamics ($\frac{dx}{dt} = v, \frac{dv}{dt} = a$). Delta modulation ($S(t) = \Theta(|x(t) - x(t-1)| - \delta_{\text{th}})$) naturally acts as an event-driven kinematic derivative filter, firing only when sudden physical anomalies or spoofing jumps occur.
  2. **Add an Encoding Energy Overhead Analysis:**
     * Include the computational cost of the encoding step itself so reviewers see the complete end-to-end pipeline budget.

---

### 📋 Summary Table: Problem vs. Solution

| Reviewer Criticism | Root Cause in Current Repo | Concrete Recommended Action |
| :--- | :--- | :--- |
| **Synthetic Data Issue** | `dataset.py` uses Gaussian distributions. | Provide real VeReMi trace parser + retain synthetic generator for stress-testing. |
| **Novelty Overstatement** | Standard SNN equations framed as new. | Reframe as domain pioneering + introduce Kinematic-Decay LIF ($\beta(t)$). |
| **Simulated Hardware Claims** | Energy/Latency computed via static multipliers. | Rephrase to "Analytical 28nm Modeling" + run wall-clock test on embedded target. |
| **Missing Quantized Baselines** | Only compared against FP32 ANN. | Add INT8 Quantized MLP to prove SNN sparsity beats quantized dense networks. |
| **Theoretical Depth** | No domain-specific mathematical proofs. | Add physical derivation connecting Newtonian kinematic residuals to spike thresholds. |
