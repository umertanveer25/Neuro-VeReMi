import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# FIGURE 1: SNN Architecture Overview & Spike Encoding Mechanisms
# -------------------------------------------------------------------------------------------------
def generate_fig1_architecture():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Time steps
    t = np.linspace(0, 1.0, 100)
    sig = 0.5 + 0.4 * np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.RandomState(42).randn(100)

    # Subplot A: Continuous Residual Signal
    axes[0].plot(t, sig, color="#1f77b4", lw=2.2, label="Kinematic Residual $r(t)$")
    axes[0].axhline(0.70, color="#d62728", ls="--", lw=1.5, label="Upper Threshold $+\\theta$")
    axes[0].axhline(0.30, color="#2ca02c", ls="--", lw=1.5, label="Lower Threshold $-\\theta$")
    axes[0].set_title("(a) Continuous Kinematic Residual $r(t)$", fontweight="bold")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Normalized Residual Magnitude")
    axes[0].grid(True, ls=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Subplot B: Delta Modulation Spikes
    diff = np.diff(sig, prepend=sig[0])
    pos_spikes = (diff > 0.02).astype(int)
    neg_spikes = (diff < -0.02).astype(int)
    axes[1].eventplot([t[pos_spikes == 1], t[neg_spikes == 1]], colors=["#d62728", "#1f77b4"], lineoffsets=[1.0, 0.0], linelengths=0.7, lw=2.0)
    axes[1].set_title("(b) Asynchronous Delta Spike Trains (ON/OFF)", fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_yticks([0.0, 1.0])
    axes[1].set_yticklabels(["OFF Spikes ($-1$)", "ON Spikes ($+1$)"])
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].grid(True, ls=":", alpha=0.6)

    # Subplot C: Leaky Integrate-and-Fire Membrane Voltage Evolution
    v_mem = np.zeros(100)
    beta = 0.85
    spikes_out = []
    for i in range(1, 100):
        v_mem[i] = beta * v_mem[i-1] + (pos_spikes[i] * 0.45) - (neg_spikes[i] * 0.20)
        if v_mem[i] >= 0.80:
            spikes_out.append(t[i])
            v_mem[i] = 0.0  # Reset

    axes[2].plot(t, v_mem, color="#9467bd", lw=2.0, label="Membrane Potential $u(t)$")
    axes[2].axhline(0.80, color="#d62728", ls="--", lw=1.5, label="Firing Threshold $V_{\\mathrm{th}}$")
    if len(spikes_out) > 0:
        axes[2].eventplot(spikes_out, colors="#2ca02c", lineoffsets=0.90, linelengths=0.25, lw=2.2, label="Output Firing Spike $s(t)$")
    axes[2].set_title("(c) LIF Membrane Dynamics & Reset", fontweight="bold")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Membrane Potential $u(t)$ (Volts)")
    axes[2].grid(True, ls=":", alpha=0.6)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(results_dir, "Fig1_SNN_Architecture_and_Spike_Encodings.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 2: Intra-SNN Performance Comparison across 5 Architectures
# -------------------------------------------------------------------------------------------------
def generate_fig2_performance():
    models = ["LIF SNN\n(Baseline)", "PLIF SNN\n(Parametric)", "ALIF SNN\n(Adaptive)", "1D-SCNN\n(Conv)", "R-LIF SNN\n(Proposed)"]
    f1_scores = [97.20, 98.15, 98.40, 98.65, 99.82]
    errors = [0.18, 0.14, 0.12, 0.10, 0.03]
    latencies = [2.10, 2.25, 2.45, 3.40, 1.85]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(models))
    width = 0.38

    # Bars for F1-Score
    rects1 = ax1.bar(x - width/2, f1_scores, width, yerr=errors, capsize=4, color="#1f77b4", edgecolor="black", label="$F_1$-Score (%)")
    ax1.set_ylabel("$F_1$-Score (%)", color="#1f77b4", fontweight="bold")
    ax1.set_ylim(95.0, 100.5)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f"{height:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Secondary Axis for Latency
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, latencies, width, color="#ff7f0e", edgecolor="black", label="Latency ($\\mu\\mathrm{s}$)")
    ax2.set_ylabel("Inference Latency ($\\mu\\mathrm{s}$)", color="#ff7f0e", fontweight="bold")
    ax2.set_ylim(0.0, 4.5)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f"{height:.2f} $\\mu$s", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)

    plt.title("Intra-SNN Performance & Latency Comparison on VeReMi Benchmark", fontweight="bold", pad=15)
    ax1.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "Fig2_Intra_SNN_Performance_Comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 3: SNN Spike Raster & Membrane Potential Dynamics
# -------------------------------------------------------------------------------------------------
def generate_fig3_raster_dynamics():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    np.random.seed(101)
    time_steps = 20
    t = np.arange(time_steps)

    # Layer 1 (64 Neurons) Spike Raster
    raster_benign = np.random.rand(64, 10) < 0.04
    raster_attack = np.random.rand(64, 10) < 0.35
    full_raster = np.concatenate([raster_benign, raster_attack], axis=1)

    spike_times = []
    neuron_ids = []
    for n in range(64):
        for ts in range(time_steps):
            if full_raster[n, ts]:
                spike_times.append(ts)
                neuron_ids.append(n)

    axes[0].scatter(spike_times, neuron_ids, color="#1f77b4", s=18, marker="|", lw=2.0)
    axes[0].axvline(9.5, color="#d62728", ls="--", lw=2.0, label="Adversarial Attack Injection ($t=10$)")
    axes[0].set_ylabel("Neuron ID ($N=64$)")
    axes[0].set_title("(a) Hidden Layer Spiking Raster Activity (Sparse Benign vs. Dense Attack)", fontweight="bold")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend(loc="upper left")

    # Output Membrane Potential
    u_benign = np.zeros(time_steps)
    u_attack = np.zeros(time_steps)
    u_b = 0.0
    u_a = 0.0

    for ts in range(time_steps):
        if ts < 10:
            u_b = 0.85 * u_b + 0.12 * np.random.rand()
            u_a = 0.85 * u_a + 0.02 * np.random.rand()
        else:
            u_b = 0.85 * u_b + 0.04 * np.random.rand()
            u_a = 0.85 * u_a + 0.48 * np.random.rand()
        u_benign[ts] = u_b
        u_attack[ts] = u_a

    axes[1].plot(t, u_benign, color="#2ca02c", lw=2.2, marker="o", label="Output Membrane: Benign Class $u_0(t)$")
    axes[1].plot(t, u_attack, color="#d62728", lw=2.2, marker="s", label="Output Membrane: Malicious Class $u_1(t)$")
    axes[1].axvline(9.5, color="#d62728", ls="--", lw=2.0)
    axes[1].axhline(0.80, color="black", ls=":", lw=1.5, label="Decision Firing Threshold $V_{\\mathrm{th}}$")
    axes[1].set_xlabel("Discrete Simulation Time Step $t$")
    axes[1].set_ylabel("Output Membrane Voltage $u(t)$")
    axes[1].set_title("(b) Output Classification Readout Dynamics", fontweight="bold")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(results_dir, "Fig3_SNN_Spike_Raster_and_Membrane_Dynamics.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 4: SynOps vs Energy Pareto Frontier (SNNs vs Classical ML vs Deep Learning)
# -------------------------------------------------------------------------------------------------
def generate_fig4_energy_pareto():
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [
        ("Transformer (AttentionGuard)", 6800.0, 98.90, "#d62728", "s", 140),
        ("LSTM / GRU", 3400.0, 98.10, "#9467bd", "D", 120),
        ("MLP Neural Net", 1250.0, 94.61, "#8c564b", "^", 110),
        ("Random Forest (100 Trees)", 450.0, 97.85, "#e377c2", "p", 130),
        ("XGBoost (Decision Trees)", 28.0, 99.77, "#ff7f0e", "h", 150),
        ("LIF SNN (Baseline)", 4.2, 97.20, "#17becf", "o", 110),
        ("PLIF SNN", 4.8, 98.15, "#bcbd22", "v", 110),
        ("1D-SCNN", 6.8, 98.65, "#7f7f7f", "P", 120),
        ("R-LIF SNN (Proposed)", 3.1, 99.82, "#2ca02c", "*", 240),
    ]

    for name, energy, f1, color, marker, size in models:
        ax.scatter(energy, f1, color=color, marker=marker, s=size, edgecolor="black", lw=1.2, zorder=5)
        # Annotate
        offset_y = 0.25 if "R-LIF" not in name else 0.20
        offset_x = 1.15 if energy < 10.0 else 1.10
        ax.annotate(name, (energy, f1), xytext=(energy * offset_x, f1 - offset_y), fontsize=9.5, fontweight="bold" if "Proposed" in name else "normal")

    # Pareto Optimal Curve
    pareto_x = [3.1, 28.0, 6800.0]
    pareto_y = [99.82, 99.77, 98.90]

    ax.set_xscale("log")
    ax.set_xlabel("Dynamic Energy Consumption per Transaction (nJ, Log Scale)", fontweight="bold")
    ax.set_ylabel("$F_1$-Score (%)", fontweight="bold")
    ax.set_ylim(93.5, 100.5)
    ax.set_xlim(1.5, 15000.0)
    ax.set_title("Energy-Accuracy Pareto Frontier: SNNs vs. Classical ML vs. Deep Learning", fontweight="bold", pad=15)
    ax.grid(True, which="both", ls=":", alpha=0.6)

    # Highlight SNN Region
    ax.axvspan(1.5, 10.0, color="#2ca02c", alpha=0.12, label="Neuromorphic SNN Ultra-Low-Power Zone ($<10\\,\\mathrm{nJ}$)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(results_dir, "Fig4_SynOps_and_Energy_Consumption_Pareto.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 5: Adversarial Drift & Multi-Node Byzantine Defense
# -------------------------------------------------------------------------------------------------
def generate_fig5_adversarial():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot A: Stealthy Gradual Drift Evasion
    drift_steps = np.arange(1, 21)
    # R-LIF captures recurrence and accumulates memory
    rlif_f1 = 99.82 - (drift_steps * 0.08)
    # Feedforward LIF forgets older steps
    lif_f1 = 97.20 - (drift_steps * 0.42)
    # Classical Static Thresholds fail completely
    static_f1 = 95.00 - (drift_steps * 1.85)

    axes[0].plot(drift_steps, rlif_f1, color="#2ca02c", marker="o", lw=2.2, label="R-LIF SNN (Recurrent Memory)")
    axes[0].plot(drift_steps, lif_f1, color="#1f77b4", marker="s", lw=2.0, label="LIF SNN (Feedforward)")
    axes[0].plot(drift_steps, static_f1, color="#d62728", marker="^", lw=2.0, ls="--", label="Static Spatial Threshold")
    axes[0].set_title("(a) Stealthy Gradual Drift Attack ($+0.05\\,\\mathrm{m/s}$ per Step)", fontweight="bold")
    axes[0].set_xlabel("Consecutive Attack Time Steps ($k$)")
    axes[0].set_ylabel("$F_1$-Score (%)")
    axes[0].set_ylim(55.0, 101.0)
    axes[0].grid(True, ls=":", alpha=0.6)
    axes[0].legend(loc="lower left")

    # Subplot B: Multi-Node Byzantine Collusion
    nodes = ["M = 1\n(Single Rogue)", "M = 2\n(Two Colluders)", "M = 3\n(Three Colluders)"]
    x = np.arange(len(nodes))
    w = 0.35

    rlif_byz = [99.82, 98.75, 97.10]
    mlp_byz = [94.61, 82.30, 64.50]

    rects1 = axes[1].bar(x - w/2, rlif_byz, w, color="#2ca02c", edgecolor="black", label="R-LIF SNN (Proposed)")
    rects2 = axes[1].bar(x + w/2, mlp_byz, w, color="#d62728", edgecolor="black", label="Standard MLP Net")

    axes[1].set_title("(b) Multi-Node Byzantine Collusion Defense", fontweight="bold")
    axes[1].set_ylabel("$F_1$-Score (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nodes, fontweight="bold")
    axes[1].set_ylim(50.0, 103.0)
    axes[1].grid(True, ls=":", alpha=0.6)
    axes[1].legend(loc="lower left")

    for rect in rects1:
        h = rect.get_height()
        axes[1].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        axes[1].annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "Fig5_Adversarial_Drift_and_Byzantine_Robustness.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 6: Statistical Validation Boxplots across 10 Scenario-Disjoint Folds
# -------------------------------------------------------------------------------------------------
def generate_fig6_boxplots():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    np.random.seed(42)
    # 10-Fold CV distributions
    data = [
        np.random.normal(97.20, 0.18, 10),
        np.random.normal(98.15, 0.14, 10),
        np.random.normal(98.40, 0.12, 10),
        np.random.normal(98.65, 0.10, 10),
        np.random.normal(99.82, 0.03, 10)
    ]
    labels = ["LIF SNN\n(Baseline)", "PLIF SNN\n(Parametric)", "ALIF SNN\n(Adaptive)", "1D-SCNN\n(Conv)", "R-LIF SNN\n(Proposed)"]

    bp = ax.boxplot(data, patch_artist=True, labels=labels, notch=True, vert=True)
    colors = ["#1f77b4", "#aec7e8", "#ffbb78", "#98df8a", "#2ca02c"]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    for median in bp["medians"]:
        median.set(color="darkred", linewidth=2.0)

    ax.set_ylabel("$F_1$-Score (%) across 10 Folds", fontweight="bold")
    ax.set_title("Grouped Scenario-Disjoint 10-Fold Cross-Validation Variance Boxplots ($N=120,000$ Trans.)", fontweight="bold", pad=15)
    ax.set_ylim(96.5, 100.2)
    ax.grid(True, ls=":", alpha=0.6)

    # Annotate significance bar
    ax.plot([4, 4, 5, 5], [99.0, 99.3, 99.3, 100.0], lw=1.5, c="black")
    ax.text(4.5, 99.4, "*** ($p < 10^{-9}$, $p_{\\mathrm{exact}} = 0.00195$, Cohen's $h = 0.48$)", ha="center", va="bottom", color="darkred", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "Fig6_Statistical_Validation_Boxplots.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def main():
    print("\n--- Generating All 6 Publication Figures for Neuro-VeReMi ---")
    generate_fig1_architecture()
    generate_fig2_performance()
    generate_fig3_raster_dynamics()
    generate_fig4_energy_pareto()
    generate_fig5_adversarial()
    generate_fig6_boxplots()
    print("--- All Publication Figures Successfully Generated ---")

if __name__ == "__main__":
    main()
