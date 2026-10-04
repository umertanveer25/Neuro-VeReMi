import torch
import torch.nn as nn
import torch.nn.functional as F

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid Surrogate Gradient for non-differentiable Heaviside step function.
    Forward: Heaviside step spike s = Theta(u - V_th)
    Backward: Derivative of sigmoid function sigma'(u) = 1 / (1 + k|u - V_th|)^2
    """
    @staticmethod
    def forward(ctx, mem, v_th=1.0, k=25.0):
        ctx.save_for_backward(mem)
        ctx.v_th = v_th
        ctx.k = k
        return (mem >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        v_th = ctx.v_th
        k = ctx.k
        grad = grad_output / (1.0 + k * torch.abs(mem - v_th)) ** 2
        return grad, None, None


def spike_fn(mem, v_th=1.0):
    return FastSigmoidSurrogate.apply(mem, v_th)


class LIF_SNN(nn.Module):
    """
    Standard Feedforward Leaky Integrate-and-Fire (LIF) SNN.
    u[t] = beta * u[t-1] * (1 - s[t-1]) + W * x[t]
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=2, beta=0.85, v_th=1.0):
        super(LIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_th = v_th

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, spike_seq):
        """
        spike_seq: (Time_Steps, Batch, Input_Dim)
        Returns: output_spikes (Time_Steps, Batch, Num_Classes), total_synops
        """
        time_steps, batch_size, _ = spike_seq.shape
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]
            # Count input synaptic events
            synops_count += inp.sum().item() * self.hidden_dim

            # Layer 1 LIF
            cur1 = self.fc1(inp)
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.hidden_dim

            # Layer 2 LIF
            cur2 = self.fc2(spk1)
            mem2 = self.beta * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, self.v_th)
            synops_count += spk2.sum().item() * self.num_classes

            # Output Readout Layer (Integrator without reset for rate coding)
            cur_out = self.fc_out(spk2)
            mem_out = self.beta * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class PLIF_SNN(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) SNN with Learnable Decay Rate.
    beta = sigmoid(w_beta) learned dynamically via backpropagation.
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=2, init_beta=0.85, v_th=1.0):
        super(PLIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.v_th = v_th

        self.w_beta1 = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_beta))))
        self.w_beta2 = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_beta))))
        self.w_beta_out = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_beta))))

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        beta1 = torch.sigmoid(self.w_beta1)
        beta2 = torch.sigmoid(self.w_beta2)
        beta_out = torch.sigmoid(self.w_beta_out)

        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
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
            synops_count += spk1.sum().item() * self.hidden_dim

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
    Firing threshold V_th[t] adapts dynamically based on neuronal spike fatigue:
    V_th[t] = V_0 + rho * a[t], where a[t] = gamma * a[t-1] + spk[t-1]
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=2, beta=0.85, v_0=1.0, rho=0.15, gamma=0.90):
        super(ALIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_0 = v_0
        self.rho = rho
        self.gamma = gamma

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        adapt1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        adapt2 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]
            synops_count += inp.sum().item() * self.hidden_dim

            # Adapt threshold for Layer 1
            adapt1 = self.gamma * adapt1 + spk1
            v_th1 = self.v_0 + self.rho * adapt1
            cur1 = self.fc1(inp)
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, v_th1)
            synops_count += spk1.sum().item() * self.hidden_dim

            # Adapt threshold for Layer 2
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
    Extracts multi-scale temporal spike correlations across temporal receptive fields.
    """
    def __init__(self, input_dim=8, num_filters=32, hidden_dim=64, num_classes=2, beta=0.85, v_th=1.0):
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
        # spike_seq: (Time_Steps, Batch, Input_Dim) -> permute to (Batch, Input_Dim, Time_Steps) for Conv1D
        time_steps, batch_size, input_dim = spike_seq.shape
        x_conv_in = spike_seq.permute(1, 2, 0)
        
        # 1D Spiking Convolutional Features
        conv_out = self.conv1(x_conv_in)  # (Batch, Num_Filters, Time_Steps)
        conv_spk = conv_out.permute(2, 0, 1)  # (Time_Steps, Batch, Num_Filters)

        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)
        spk1 = torch.zeros_like(mem1)

        out_spikes = []
        synops_count = spike_seq.sum().item() * self.num_filters * 3  # Kernel size 3

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
    Proposed Core Model: Recurrent Leaky Integrate-and-Fire (R-LIF) SNN.
    Incorporates internal inter-neuronal recurrent synaptic connections V_rec:
    u[t] = beta * u[t-1] * (1 - s[t-1]) + W_in * x[t] + V_rec * s[t-1]
    Enables native long-term kinematic memory across sequential vehicular BSM transmissions.
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=2, beta=0.85, v_th=1.0):
        super(RLIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_th = v_th

        # Feedforward Synapses
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        # Recurrent Feedback Synapses
        self.v_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Layer 2 Feedforward
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output Readout Layer
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        # Initialize recurrent weights orthogonal for stable gradients
        nn.init.orthogonal_(self.v_rec.weight, gain=0.8)

    def forward(self, spike_seq):
        time_steps, batch_size, _ = spike_seq.shape
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem2 = torch.zeros(batch_size, self.hidden_dim, device=spike_seq.device)
        mem_out = torch.zeros(batch_size, self.num_classes, device=spike_seq.device)

        spk1 = torch.zeros_like(mem1)
        spk2 = torch.zeros_like(mem2)

        out_spikes = []
        synops_count = 0

        for t in range(time_steps):
            inp = spike_seq[t]
            synops_count += inp.sum().item() * self.hidden_dim

            # Layer 1: Feedforward + Recurrent Feedback
            cur1 = self.fc_in(inp) + self.v_rec(spk1)
            synops_count += spk1.sum().item() * self.hidden_dim  # Recurrent SynOps
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.hidden_dim

            # Layer 2 Feedforward
            cur2 = self.fc2(spk1)
            mem2 = self.beta * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, self.v_th)
            synops_count += spk2.sum().item() * self.num_classes

            # Output Readout Layer
            cur_out = self.fc_out(spk2)
            mem_out = self.beta * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count
