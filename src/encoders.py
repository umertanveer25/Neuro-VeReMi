import torch
import torch.nn as nn
import numpy as np

class DeltaModulationEncoder:
    """
    Asynchronous Kinematic Delta Modulation Spike Encoder.
    Maps 7-dimensional continuous kinematic telemetry residuals into a 14-channel
    bipolar spike event vector: S_in(t) in {0, 1}^14 (7 ON channels, 7 OFF channels).
    Emits spikes when deviations exceed differential thresholds +/- theta_m:
      S+_{k,m}(t) = Theta( Delta r_{k,m}(t) - theta_m )
      S-_{k,m}(t) = Theta( -Delta r_{k,m}(t) - theta_m )
    """
    def __init__(self, threshold=0.08, num_steps=10):
        self.threshold = threshold
        self.num_steps = num_steps

    def encode(self, x_seq):
        """
        x_seq: Tensor of shape (Batch, 7) or (Batch, Seq_Len, 7)
        Returns: Spike tensor of shape (Time_Steps, Batch, 14)
        """
        if x_seq.dim() == 2:
            batch_size, num_features = x_seq.shape
            x_time = x_seq.unsqueeze(1).repeat(1, self.num_steps, 1)
            # Add temporal dynamic trend modeling car-following trajectory evolution
            trend = torch.linspace(0.92, 1.0, self.num_steps).unsqueeze(0).unsqueeze(2).to(x_seq.device)
            x_time = x_time * trend
        else:
            x_time = x_seq
            batch_size, self.num_steps, num_features = x_time.shape

        # Compute temporal derivative dx/dt across successive time steps
        diff = torch.zeros_like(x_time)
        diff[:, 1:, :] = x_time[:, 1:, :] - x_time[:, :-1, :]
        diff[:, 0, :] = x_time[:, 0, :] * 0.15  # Initial step boundary condition

        # Emits dual-rail ON (positive jerk/deviation) and OFF (negative) spikes
        pos_spikes = (diff > self.threshold).float()
        neg_spikes = (diff < -self.threshold).float()

        # Concatenate ON and OFF channels -> shape (Batch, Time_Steps, 2 * num_features) = (Batch, Time_Steps, 14)
        spikes = torch.cat([pos_spikes, neg_spikes], dim=-1)
        # Permute to (Time_Steps, Batch, 14) for temporal SNN execution
        return spikes.permute(1, 0, 2)


class PoissonRateEncoder:
    """
    Poisson Rate Spike Encoder.
    Maps continuous normalized feature intensities into stochastic Bernoulli spike trains.
    Produces dual-rail 14-channel output for architecture compatibility.
    """
    def __init__(self, num_steps=10, max_rate=0.85):
        self.num_steps = num_steps
        self.max_rate = max_rate

    def encode(self, x):
        """
        x: Tensor of shape (Batch, 7)
        Returns: Spike tensor of shape (Time_Steps, Batch, 14)
        """
        batch_size, num_features = x.shape
        x_pos = torch.clamp(x, min=0.0)
        x_neg = torch.clamp(-x, min=0.0)
        x_dual = torch.cat([x_pos, x_neg], dim=-1)  # (Batch, 14)
        
        norm_rate = torch.clamp(x_dual * 0.5, 0.01, self.max_rate)
        # Expand over time steps
        expanded_rate = norm_rate.unsqueeze(0).repeat(self.num_steps, 1, 1)
        rand_tensor = torch.rand_like(expanded_rate)
        spikes = (rand_tensor < expanded_rate).float()
        return spikes


class LatencyEncoder:
    """
    Time-to-First-Spike (TTFS) Latency Spike Encoder.
    Higher continuous values trigger spikes earlier in the temporal evaluation window.
    Produces dual-rail 14-channel output for architecture compatibility.
    """
    def __init__(self, num_steps=10, tau=1.0):
        self.num_steps = num_steps
        self.tau = tau

    def encode(self, x):
        """
        x: Tensor of shape (Batch, 7)
        Returns: Spike tensor of shape (Time_Steps, Batch, 14)
        """
        batch_size, num_features = x.shape
        x_pos = torch.clamp(x, min=0.0)
        x_neg = torch.clamp(-x, min=0.0)
        x_dual = torch.cat([x_pos, x_neg], dim=-1)  # (Batch, 14)

        norm_val = torch.clamp(x_dual * 0.5, 0.001, 1.0)
        t_spikes = torch.clamp(((1.0 - norm_val) * (self.num_steps - 1)).long(), 0, self.num_steps - 1)

        spikes = torch.zeros((self.num_steps, batch_size, 14), device=x.device)
        for t in range(self.num_steps):
            spikes[t] = (t_spikes == t).float()
        return spikes
