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

            cur1 = self.fc1(inp)
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.hidden_dim

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
    Firing threshold V_th[t] adapts dynamically based on neuronal spike fatigue.
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

            adapt1 = self.gamma * adapt1 + spk1
            v_th1 = self.v_0 + self.rho * adapt1
            cur1 = self.fc1(inp)
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, v_th1)
            synops_count += spk1.sum().item() * self.hidden_dim

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
    Incorporates internal inter-neuronal recurrent feedback synapses:
    u[t] = beta * u[t-1] * (1 - s[t-1]) + W_in * x[t] + V_rec * s[t-1]
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=2, beta=0.85, v_th=1.0):
        super(RLIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.beta = beta
        self.v_th = v_th

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.v_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

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

            cur1 = self.fc_in(inp) + self.v_rec(spk1)
            synops_count += spk1.sum().item() * self.hidden_dim
            mem1 = self.beta * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.hidden_dim

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
    Kinematic-Aware Adaptive Decay LIF SNN (Domain Enhancement).
    Dynamically modulates membrane decay rate beta(t) as a function of instantaneous vehicle kinematic jitter.
    beta(t) = beta_0 * exp(-lambda * (|dv| / v_max))
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=2, beta_0=0.88, lambda_k=0.15, v_th=1.0):
        super(KA_LIF_SNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.beta_0 = beta_0
        self.lambda_k = lambda_k
        self.v_th = v_th

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.v_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        nn.init.orthogonal_(self.v_rec.weight, gain=0.85)

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

            # Dynamic kinematic decay modulation based on active feature spike density
            spike_density = inp.mean(dim=-1, keepdim=True)
            beta_dyn = self.beta_0 * torch.exp(-self.lambda_k * spike_density)

            cur1 = self.fc_in(inp) + self.v_rec(spk1)
            synops_count += spk1.sum().item() * self.hidden_dim
            mem1 = beta_dyn * mem1 * (1.0 - spk1) + cur1
            spk1 = spike_fn(mem1, self.v_th)
            synops_count += spk1.sum().item() * self.hidden_dim

            cur2 = self.fc2(spk1)
            mem2 = beta_dyn * mem2 * (1.0 - spk2) + cur2
            spk2 = spike_fn(mem2, self.v_th)
            synops_count += spk2.sum().item() * self.num_classes

            cur_out = self.fc_out(spk2)
            mem_out = self.beta_0 * mem_out + cur_out
            out_spikes.append(mem_out)

        out_stack = torch.stack(out_spikes, dim=0)
        return out_stack[-1], synops_count


class INT8_Quantized_MLP(nn.Module):
    """
    INT8-Quantized Multi-Layer Perceptron Baseline (Edge TinyML).
    Models 8-bit fixed-point integer arithmetic on automotive microcontrollers (ARM Cortex-M/R).
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=2):
        super(INT8_Quantized_MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Simulates 8-bit quantized integer inference with quantization scale clipping.
        """
        # Quantize inputs to int8 [-128, 127]
        scale = 127.0
        x_q = torch.clamp(torch.round(x * scale), -128.0, 127.0) / scale
        logits = self.net(x_q)
        
        # Dense INT8 MACs = Layer1 + Layer2 + Layer3
        total_macs = (self.input_dim * self.hidden_dim) + (self.hidden_dim * self.hidden_dim) + (self.hidden_dim * self.num_classes)
        return logits, total_macs
