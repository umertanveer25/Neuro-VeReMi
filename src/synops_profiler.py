import torch
import numpy as np

class NeuromorphicProfiler:
    """
    Neuromorphic Hardware & Energy Profiler.
    Models energy consumption, Synaptic Operations (SynOps), and embedded execution latency:
    - E_AC (Accumulate Operation in 45nm/28nm SNN): 0.9 pJ
    - E_MAC (Multiply-Accumulate in 45nm/28nm ANN): 4.6 pJ
    - Static Leakage Energy per neuron: 0.1 pJ / timestep
    """
    def __init__(self, e_ac_pj=0.9, e_mac_pj=4.6, e_leak_pj=0.1):
        self.e_ac_pj = e_ac_pj
        self.e_mac_pj = e_mac_pj
        self.e_leak_pj = e_leak_pj

    def profile_snn(self, model, spike_encoder, x_sample, time_steps=10):
        """
        Profiles a single transaction inference through the SNN.
        Returns: SynOps, Energy (nJ), Latency (us), Sparsity (%)
        """
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)
            spikes = spike_encoder.encode(x_tensor)
            
            # Compute Input Spike Sparsity
            total_elements = spikes.numel()
            active_spikes = spikes.sum().item()
            input_sparsity = 1.0 - (active_spikes / (total_elements + 1e-8))

            # Run inference and count SynOps
            out, synops = model(spikes)

            # Total Energy (nJ) = (SynOps * E_AC + N_neurons * T * E_leak) / 1000.0
            num_hidden = getattr(model, 'hidden_dim', 64)
            leakage_pj = (num_hidden * 2 + 2) * time_steps * self.e_leak_pj
            total_energy_pj = (synops * self.e_ac_pj) + leakage_pj
            total_energy_nj = total_energy_pj / 1000.0

            # Modeled Embedded Latency on ARM Cortex-R52 (400 MHz) with TCM:
            # Each SynOp is an integer addition (1-2 cycles), plus pipeline overhead
            cycles = synops * 1.2 + 150
            latency_us = cycles / 400.0  # 400 MHz clock

        return {
            "synops": synops,
            "energy_nj": total_energy_nj,
            "latency_us": latency_us,
            "input_sparsity_pct": input_sparsity * 100.0
        }

    def profile_dense_ann(self, input_dim=8, hidden_dim=64, num_classes=2, num_layers=3):
        """
        Profiles standard dense ANN (MLP/DNN) for comparison.
        """
        # Dense MACs = Layer1 + Layer2 + Out
        macs = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + (hidden_dim * num_classes)
        energy_pj = macs * self.e_mac_pj
        energy_nj = energy_pj / 1000.0
        cycles = macs * 8.0 + 300  # FP32 multiplications require ~8 cycles
        latency_us = cycles / 400.0

        return {
            "macs": macs,
            "energy_nj": energy_nj,
            "latency_us": latency_us,
            "sparsity_pct": 0.0
        }
