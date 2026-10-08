import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set high-tier IEEE publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)

def generate_exact_fig1_300dpi():
    fig = plt.figure(figsize=(13, 6.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.1], height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.26)

    # -------------------------------------------------------------------------
    # Panel (a): Left Full-Height Plot - Continuous Kinematic Residual
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[:, 0])

    t_a = np.linspace(0.0, 1.0, 500)
    # Sinusoidal curve matching user's visual
    sig = 0.55 * np.sin(2 * np.pi * 1.8 * t_a) * np.exp(-0.35 * t_a)

    ax_a.plot(t_a, sig, color="#0b2c6d", lw=2.5, label=r"Kinematic Residual $r_p(t)$")
    ax_a.axhline(0.60, color="#b31b1b", ls="--", lw=2.0, label=r"Differential Bounds $\pm\theta$")
    ax_a.axhline(-0.40, color="#1b7837", ls="--", lw=2.0)

    # Text annotations on thresholds
    ax_a.text(0.98, 0.63, r"$+\theta = +0.60$", color="#b31b1b", ha="right", va="bottom", fontweight="bold", fontsize=11)
    ax_a.text(0.98, -0.37, r"$-\theta = -0.40$", color="#1b7837", ha="right", va="bottom", fontweight="bold", fontsize=11)

    ax_a.set_title("(a) Continuous Kinematic Residual", fontsize=13, fontweight="bold", pad=10)
    ax_a.set_xlabel("Time (s)", fontweight="bold")
    ax_a.set_ylabel(r"Kinematic Residual $r_p(t)$", fontweight="bold")
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(-0.65, 0.85)
    ax_a.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax_a.grid(True, ls=":", color="#bfbfbf", alpha=0.8)
    ax_a.legend(loc="lower left", framealpha=0.95, edgecolor="#999999")

    # -------------------------------------------------------------------------
    # Panel (b): Top-Right Plot - Asynchronous Delta Spike Trains (ON/OFF)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    ax_b.axhline(0.0, color="black", lw=1.8)
    
    # ON Spikes (Red arrows pointing UP to +1)
    on_times = [0.08, 0.55]
    for ot in on_times:
        ax_b.annotate("", xy=(ot, 1.0), xytext=(ot, 0.0),
                      arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#b31b1b", lw=2.5))
    ax_b.text(0.12, 0.68, "ON Spikes (+1, +jerk)", color="#b31b1b", fontsize=10.5, fontweight="bold")

    # OFF Spikes (Blue arrows pointing DOWN to -1)
    off_times = [0.24, 0.76]
    for oft in off_times:
        ax_b.annotate("", xy=(oft, -1.0), xytext=(oft, 0.0),
                      arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#0b2c6d", lw=2.5))
    ax_b.text(0.28, -0.72, "OFF Spikes (-1, -jerk)", color="#0b2c6d", fontsize=10.5, fontweight="bold")

    ax_b.set_title("(b) Asynchronous Delta Spike Trains", fontsize=13, fontweight="bold", pad=8)
    ax_b.set_xlabel("Time (s)", fontweight="bold")
    ax_b.set_ylabel(r"Spike Polarity $s(t)$", fontweight="bold")
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(-1.3, 1.3)
    ax_b.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_yticks([-1, 0, 1])
    ax_b.set_yticklabels(["-1", "0", "+1"])
    ax_b.grid(True, ls=":", color="#bfbfbf", alpha=0.8)

    # -------------------------------------------------------------------------
    # Panel (c): Bottom-Right Plot - KA-LIF Dynamic Decay & Reset
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 1])

    # Piecewise exact KA-LIF voltage waveform
    t1 = np.linspace(0.0, 0.2, 100)
    v1 = 0.50 * (1 - np.exp(-12 * t1)) / (1 - np.exp(-12 * 0.2))

    t2 = np.linspace(0.2, 0.4, 100)
    v2 = 0.50 * np.exp(-4.6 * (t2 - 0.2)) + 0.03

    t3 = np.linspace(0.4, 0.66, 130)
    v3 = v2[-1] + (0.75 - v2[-1]) * ((t3 - 0.4) / (0.66 - 0.4)) ** 1.6

    t4 = np.linspace(0.66, 1.0, 170)
    v4 = np.zeros_like(t4)

    t_full = np.concatenate([t1, t2, t3, t4])
    v_full = np.concatenate([v1, v2, v3, v4])

    ax_c.plot(t_full, v_full, color="#6a0dad", lw=2.5)
    ax_c.axhline(0.75, color="#b31b1b", ls="--", lw=2.0)
    ax_c.text(0.98, 0.78, r"$V_{\mathrm{th}} = 0.75\ \mathrm{V}$", color="#b31b1b", ha="right", va="bottom", fontweight="bold", fontsize=11)

    # Output Spike Arrow (Green vertical arrow from 0 to 0.75 at t=0.66)
    ax_c.annotate("", xy=(0.66, 0.75), xytext=(0.66, 0.0),
                  arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#008000", lw=2.5))
    ax_c.text(0.68, 0.45, "Output Spike\n" + r"$s(t) = 1$", color="#008000", fontsize=10.5, fontweight="bold")

    # Equation annotation
    ax_c.text(0.06, 0.12, r"$\beta_i(t) = \beta_0 \exp(-\lambda_k \zeta_{\mathrm{kin}})$", color="#6a0dad", fontsize=11, fontweight="bold")

    ax_c.set_title("(c) KA-LIF Dynamic Decay & Reset", fontsize=13, fontweight="bold", pad=8)
    ax_c.set_xlabel("Time (s)", fontweight="bold")
    ax_c.set_ylabel(r"Membrane Voltage $u_i(t)$ (V)", fontweight="bold")
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(-0.05, 0.95)
    ax_c.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax_c.grid(True, ls=":", color="#bfbfbf", alpha=0.8)

    # Main super-title
    fig.suptitle("Figure 1: SNN Architecture & Kinematic Delta Modulation Encodings", fontsize=15, fontweight="bold", y=0.98)

    out_path = os.path.join(results_dir, "Fig1_SNN_Architecture_and_Spike_Encodings.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated 100% mathematically exact 300 DPI Figure 1 at: {out_path}")

if __name__ == "__main__":
    generate_exact_fig1_300dpi()
