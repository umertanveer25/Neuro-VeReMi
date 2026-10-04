import torch
import torch.nn as nn
import numpy as np

class DeltaModulationEncoder:
    """
    Delta Modulation (Threshold) Spike Encoder.
    Emits a spike when the temporal derivative / rate-of-change of a feature exceeds threshold theta.
    Generates ternary/signed binary spike trains s in {-1, 0, 1} or dual-channel positive/negative spikes.
    """
    def __init__(self, threshold=0.08, num_steps=10):
        self.threshold = threshold
        self.num_steps = num_steps

    def encode(self, x_seq):
        """
        x_seq: Tensor of shape (Batch, Seq_Len, Features) or (Batch, Features)
        Returns: Spike tensor of shape (Time_Steps, Batch, 2 * Features) (ON/OFF channels)
        """
        if x_seq.dim() == 2:
            # Replicate over time with linear interpolation / temporal decay
            batch_size, num_features = x_seq.shape
            x_time = x_seq.unsqueeze(1).repeat(1, self.num_steps, 1)
            # Add subtle temporal trend based on kinematic momentum
            trend = torch.linspace(0.85, 1.0, self.num_steps).unsqueeze(0).unsqueeze(2).to(x_seq.device)
            x_time = x_time * trend
        else:
            x_time = x_seq
            batch_size, self.num_steps, num_features = x_time.shape

        # Compute temporal derivative dx/dt
        diff = torch.zeros_like(x_time)
        diff[:, 1:, :] = x_time[:, 1:, :] - x_time[:, :-1, :]
        diff[:, 0, :] = x_time[:, 0, :] * 0.1  # Initial boundary spike

        # Separate into positive (ON) and negative (OFF) spike channels
        pos_spikes = (diff > self.threshold).float()
        neg_spikes = (diff < -self.threshold).float()

        # Concatenate ON and OFF channels: shape (Batch, Time_Steps, 2 * Features)
        spikes = torch.cat([pos_spikes, neg_spikes], dim=-1)
        # Permute to (Time_Steps, Batch, 2 * Features) for SNN ingestion
        return spikes.permute(1, 0, 2)


class PoissonRateEncoder:
    """
    Poisson Rate Coding.
    Maps continuous normalized feature intensities into stochastic Bernoulli spike trains.
    """
    def __init__(self, num_steps=10, max_rate=0.85):
        self.num_steps = num_steps
        self.max_rate = max_rate

    def encode(self, x):
        """
        x: Tensor of shape (Batch, Features) with values normalized to [0, 1]
        Returns: Spike tensor of shape (Time_Steps, Batch, Features)
        """
        batch_size, num_features = x.shape
        x_norm = torch.clamp(torch.abs(x), 0.0, 1.0) * self.max_rate
        # Expand over time steps
        x_expanded = x_norm.unsqueeze(0).repeat(self.num_steps, 1, 1)
        # Stochastic Bernoulli trial
        rand_tensor = torch.rand_like(x_expanded)
        spikes = (rand_tensor < x_expanded).float()
        return spikes


class LatencyEncoder:
    """
    Latency / Time-to-First-Spike (TTFS) Coding.
    Higher continuous values fire a spike earlier in the temporal window.
    """
    def __init__(self, num_steps=10, tau=1.0):
        self.num_steps = num_steps
        self.tau = tau

    def encode(self, x):
        """
        x: Tensor of shape (Batch, Features) with values in [0, 1]
        Returns: Spike tensor of shape (Time_Steps, Batch, Features)
        """
        batch_size, num_features = x.shape
        x_norm = torch.clamp(torch.abs(x), 0.001, 1.0)
        # Compute exact spike arrival timestep: t_spike = floor((1 - x) * (num_steps - 1))
        t_spikes = torch.clamp(((1.0 - x_norm) * (self.num_steps - 1)).long(), 0, self.num_steps - 1)

        spikes = torch.zeros((self.num_steps, batch_size, num_features), device=x.device)
        for t in range(self.num_steps):
            mask = (t_spikes == t).float()
            spikes[t] = mask
        return spikes
