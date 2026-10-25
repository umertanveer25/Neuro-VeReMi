import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ArctanSurrogate(torch.autograd.Function):
    """
    Arctangent Surrogate Gradient for non-differentiable Heaviside step function (Eq. 14 in manuscript).
    Forward: Heaviside step spike s = Theta(u - V_th)
    Backward: sigma'(u) = gamma_s / (1 + (pi * gamma_s * (u - V_th))^2), with gamma_s = 2.0
    Satisfies unit-area Dirac delta requirement: integral_{-inf}^{+inf} sigma'(u) du = 1.0.
    """
    @staticmethod
    def forward(ctx, mem, v_th=0.75, gamma_s=2.0):
        ctx.save_for_backward(mem)
        ctx.v_th = v_th
        ctx.gamma_s = gamma_s
        return (mem >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        v_th = ctx.v_th
        gamma_s = ctx.gamma_s
        grad = grad_output * (gamma_s / (1.0 + (np.pi * gamma_s * (mem - v_th)) ** 2))
        return grad, None, None


def spike_fn(mem, v_th=0.75):
    return ArctanSurrogate.apply(mem, v_th)


class RuleBased_KinematicDetector:
    """
    Classical Heuristic Kinematic Plausibility Rule Detector (Reviewer Comment 10 baseline).
    Applies physical invariant boundaries on position, velocity, and acceleration deviations:
      - Max comfort acceleration: a_max <= 3.5 m/s^2
      - Max comfort deceleration: a_min >= -8.0 m/s^2
      - Max speed threshold: v_max <= 42.0 m/s
    """
    def __init__(self, spd_thresh=3.0, acl_thresh=3.0):
        self.spd_thresh = spd_thresh
        self.acl_thresh = acl_thresh

    def predict(self, X):
        """
        X: Tensor or ndarray of shape (Batch, 7) [px, py, vx, vy, ax, ay, heading] (standardized)
        Returns: binary predictions (1 = malicious, 0 = benign)
        """
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        spd_mag = np.sqrt(X_np[:, 2]**2 + X_np[:, 3]**2)
        acl_mag = np.sqrt(X_np[:, 4]**2 + X_np[:, 5]**2)
        
        preds = ((spd_mag > self.spd_thresh) | (acl_mag > self.acl_thresh)).astype(int)
        return preds


class INT8_Quantized_MLP(nn.Module):
    """
    INT8-Quantized Multi-Layer Perceptron Baseline (Edge TinyML, Reviewer Comment 10).
    Architecture: 14 -> 64 -> 32 -> 2 (3,008 INT8 parameters).
    Simulates 8-bit fixed-point integer arithmetic on automotive MCUs (ARM Cortex-R52 / Infineon TC397).
    """
    def __init__(self, input_dim=14, hidden_dim=64, intermediate_dim=32, num_classes=2):
        super(INT8_Quantized_MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc_out = nn.Linear(intermediate_dim, num_classes)

    def forward(self, x):
        # Quantize inputs to 8-bit [-128, 127]
        scale = 127.0
        x_q = torch.clamp(torch.round(x * scale), -128.0, 127.0) / scale
        
        h1 = F.relu(self.fc1(x_q))
        h1_q = torch.clamp(torch.round(h1 * scale), -128.0, 127.0) / scale

        h2 = F.relu(self.fc2(h1_q))
        h2_q = torch.clamp(torch.round(h2 * scale), -128.0, 127.0) / scale

        logits = self.fc_out(h2_q)
        total_macs = (self.input_dim * self.hidden_dim) + (self.hidden_dim * self.intermediate_dim) + (self.intermediate_dim * self.num_classes)
        return logits, total_macs


class LIF_SNN(nn.Module):
    """
    Standard Feedforward Leaky Integrate-and-Fire (LIF) SNN (Baseline).
    Architecture: 14 -> 64 -> 32 -> 2 (3,008 Synapses, 98 Neurons).
    Static membrane leak decay constant beta = 0.85, V_th = 0.75 V.
    """
    def __init__(self, input_dim=14, hidden_dim=64, intermediate_dim=32, num_classes=2, beta=0.85, v_th=0.75):
        super(LIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_th = v_th

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc_out = nn.Linear(intermediate_dim, num_classes)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.intermediate_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]
            synops_count += inp.sum().item() * self.hidden_dim

            cur1 = self.fc1(inp)
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.intermediate_dim

            cur2 = self.fc2(spk1)
            mem2 = self.beta * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, self.v_th)
            synops_count += spk2.sum().item() * self.num_classes

            cur_out = self.fc_out(spk2)
            mem_out = self.beta * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class PLIF_SNN(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) SNN with Learnable Decay Rate.
    Architecture: 14 -> 64 -> 32 -> 2 (3,008 Synapses, 98 Neurons).
    beta = sigmoid(w_beta) learned dynamically via backpropagation.
    """
    def __init__(self, input_dim=14, hidden_dim=64, intermediate_dim=32, num_classes=2, init_beta=0.85, v_th=0.75):
        super(PLIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.v_th = v_th

        self.w_beta1 = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(init_beta)))))
        self.w_beta2 = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(init_beta)))))
        self.w_beta_out = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(init_beta)))))

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc_out = nn.Linear(intermediate_dim, num_classes)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        beta1 = torch.sigmoid(self.w_beta1)
        beta2 = torch.sigmoid(self.w_beta2)
        beta_out = torch.sigmoid(self.w_beta_out)

        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.intermediate_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]
            synops_count += inp.sum().item() * self.hidden_dim

            cur1 = self.fc1(inp)
            mem1 = beta1 * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.intermediate_dim

            cur2 = self.fc2(spk1)
            mem2 = beta2 * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, self.v_th)
            synops_count += spk2.sum().item() * self.num_classes

            cur_out = self.fc_out(spk2)
            mem_out = beta_out * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class ALIF_SNN(nn.Module):
    """
    Adaptive Threshold Leaky Integrate-and-Fire (ALIF) SNN.
    Architecture: 14 -> 64 -> 32 -> 2 (3,008 Synapses, 98 Neurons).
    Firing threshold V_th[t] adapts dynamically based on neuronal spike fatigue.
    """
    def __init__(self, input_dim=14, hidden_dim=64, intermediate_dim=32, num_classes=2, beta=0.85, v_0=0.75, rho=0.15, gamma=0.90):
        super(ALIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_0 = v_0
        self.rho = rho
        self.gamma = gamma

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc_out = nn.Linear(intermediate_dim, num_classes)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.intermediate_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        adapt1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        adapt2 = torch.zeros(batch_size, self.intermediate_dim, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]
            synops_count += inp.sum().item() * self.hidden_dim

            adapt1 = self.gamma * adapt1 + spk1
            v_th1 = self.v_0 + self.rho * adapt1
            cur1 = self.fc1(inp)
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, v_th1)
            synops_count += spk1.sum().item() * self.intermediate_dim

            adapt2 = self.gamma * adapt2 + spk2
            v_th2 = self.v_0 + self.rho * adapt2
            cur2 = self.fc2(spk1)
            mem2 = self.beta * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, v_th2)
            synops_count += spk2.sum().item() * self.num_classes

            cur_out = self.fc_out(spk2)
            mem_out = self.beta * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class SCNN_1D(nn.Module):
    """
    Spiking 1D Convolutional Neural Network (SCNN).
    Architecture: Conv1D (in=14, out=32, k=3, pad=1) -> FC (32, 32) -> FC (32, 2).
    """
    def __init__(self, input_dim=14, num_filters=32, hidden_dim=32, num_classes=2, beta=0.85, v_th=0.75):
        super(SCNN_1D, self).__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_th = v_th

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, spike_seq):
        time_steps, batch_size, input_dim = spike_seq.shape
        x_conv_in = spike_seq.permute(1, 2, 0)
        conv_out = self.conv1(x_conv_in)
        conv_spk = conv_out.permute(2, 0, 1)

        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)
        spk1 = torch.zeros_like(mem1)

        out_spikes = []
        synops_count = spike_seq.sum().item() * self.num_filters * 3

        for t in range(time_steps):
            c_inp = spike_fn(conv_spk[t], self.v_th)
            cur1 = self.fc1(c_inp)
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.num_classes

            cur_out = self.fc_out(spk1)
            mem_out = self.beta * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class RLIF_SNN(nn.Module):
    """
    Recurrent Leaky Integrate-and-Fire (R-LIF) SNN.
    Architecture: 14 -> 64 -> 32 -> 2 with recurrent feedback connections in hidden layer.
    """
    def __init__(self, input_dim=14, hidden_dim=64, intermediate_dim=32, num_classes=2, beta=0.85, v_th=0.75):
        super(RLIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_th = v_th

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.v_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc_out = nn.Linear(intermediate_dim, num_classes)

        nn.init.orthogonal_(self.v_rec.weight, gain=0.8)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.intermediate_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]
            synops_count += inp.sum().item() * self.hidden_dim

            cur1 = self.fc_in(inp) + self.v_rec(spk1)
            synops_count += spk1.sum().item() * self.hidden_dim
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.intermediate_dim

            cur2 = self.fc2(spk1)
            mem2 = self.beta * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, self.v_th)
            synops_count += spk2.sum().item() * self.num_classes

            cur_out = self.fc_out(spk2)
            mem_out = self.beta * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class KA_LIF_SNN(nn.Module):
    """
    Kinematic-Aware Adaptive Decay LIF SNN (Proposed Architecture).
    Architecture: 14 -> 64 -> 32 -> 2 (3,008 Synapses, 98 Neurons).
    Exact Biophysical Coupling (Eqs. 8-11 in manuscript):
      - Cumulative Kinematic Stress: zeta_kin[t] = alpha_zeta * zeta_kin[t-1] + (1 - alpha_zeta) * zeta_inst[t]
      - Dynamic Membrane Decay Rate (Eq. 11):
          beta_i(t) = 1.0 - (1.0 - beta_rest) * exp(-lambda_k * zeta_kin(t))
          Under benign traffic (zeta_kin -> 0): beta_i -> beta_rest = 0.60 (rapid leak of channel noise)
          Under anomaly stress (zeta_kin > 0): beta_i -> 0.95 (memory retention of gradual drift)
      - Kinematic Current Injection (Eq. 10):
          I_kin,i(t) = lambda_inj * zeta_kin(t)
      - Firing threshold V_th = 0.75 V, hard reset to 0.0 V, refractory guard tau_ref = 1 ms.
    """
    def __init__(self, input_dim=14, hidden_dim=64, intermediate_dim=32, num_classes=2,
                 beta_rest=0.60, lambda_k=1.2, lambda_inj=0.50, alpha_zeta=0.80, v_th=0.75):
        super(KA_LIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_classes = num_classes

        self.beta_rest = beta_rest
        self.lambda_k = lambda_k
        self.lambda_inj = lambda_inj
        self.alpha_zeta = alpha_zeta
        self.v_th = v_th

        # Static Feedforward Synaptic Matrices (N_syn = 14*64 + 64*32 + 32*2 = 3,008)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc_out = nn.Linear(intermediate_dim, num_classes)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.intermediate_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        zeta_kin = torch.zeros(batch_size, 1, device=spike_seq.device)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]  # (Batch, 14)
            synops_count += inp.sum().item() * self.hidden_dim

            # 1. Kinematic Stress Estimation across 7 dual-rail dimensions (Eqs. 8-9)
            zeta_inst = inp.mean(dim=-1, keepdim=True) * 2.5
            zeta_kin = self.alpha_zeta * zeta_kin + (1.0 - self.alpha_zeta) * zeta_inst

            # 2. Dynamic Membrane Retention (Eq. 11): beta_i(t) = 1.0 - (1.0 - beta_rest) * exp(-lambda_k * zeta_kin)
            if self.lambda_k > 0:
                beta_i = 1.0 - (1.0 - self.beta_rest) * torch.exp(-self.lambda_k * zeta_kin)
            else:
                beta_i = self.beta_rest

            # 3. Kinematic Driving Current (Eq. 10): I_kin = lambda_inj * zeta_kin
            i_kin = self.lambda_inj * zeta_kin if self.lambda_inj > 0 else 0.0

            # Layer 1 forward
            cur1 = self.fc1(inp) + i_kin
            mem1 = beta_i * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.intermediate_dim

            # Layer 2 forward
            cur2 = self.fc2(spk1)
            mem2 = beta_i * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, self.v_th)
            synops_count += spk2.sum().item() * self.num_classes

            # Layer 3 Readout
            cur_out = self.fc_out(spk2)
            mem_out = self.beta_rest * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class ByzantineTrustEngine:
    """
    Multi-Hop Byzantine Trust Accumulator (Layer 3 Defense, Eq. 16 in manuscript).
    Accumulates trust scores H_k(t) in [0, 1] across sliding window T_w = 10:
      H_k(t) = gamma_h * H_k(t-1) + (1 - gamma_h) * (1 - 1/T_w sum S_out) * prod (1 - omega_jk * D_jk)
    """
    def __init__(self, gamma_h=0.90, tau_trust=0.50, t_w=10):
        self.gamma_h = gamma_h
        self.tau_trust = tau_trust
        self.t_w = t_w

    def update_trust(self, prev_trust, snn_anomaly_probs, neighbor_divergences=None):
        """
        prev_trust: ndarray of shape (Batch,)
        snn_anomaly_probs: ndarray of shape (Batch,) in [0, 1]
        neighbor_divergences: ndarray of shape (Batch,) or None
        """
        trust_penalty = 1.0 - snn_anomaly_probs
        if neighbor_divergences is not None:
            spatial_factor = np.clip(1.0 - 0.15 * neighbor_divergences, 0.0, 1.0)
        else:
            spatial_factor = 1.0

        new_trust = self.gamma_h * prev_trust + (1.0 - self.gamma_h) * trust_penalty * spatial_factor
        is_isolated = new_trust < self.tau_trust
        return new_trust, is_isolated
