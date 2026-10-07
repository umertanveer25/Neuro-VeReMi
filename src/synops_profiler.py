import torch
import numpy as np

class NeuromorphicProfiler:
    """
    Neuromorphic Hardware & Energy Profiler.
    Models energy consumption, Synaptic Operations (SynOps), and embedded execution latency:
    - E_AC (Accumulate Operation in 28nm SNN): 0.9 pJ
    - E_MAC_FP32 (Multiply-Accumulate in 28nm ANN): 4.6 pJ
    - E_MAC_INT8 (Quantized 8-bit MAC in 28nm TinyML): 0.2 pJ
    - E_ENC (Delta Modulation Encoder ALU comparison): 0.1 pJ / feature
    - E_LEAK (Static Leakage Energy per neuron): 0.1 pJ / timestep
    """
    def __init__(self, e_ac_pj=0.9, e_mac_fp32_pj=4.6, e_mac_int8_pj=0.2, e_enc_pj=0.1, e_leak_pj=0.1):
        self.e_ac_pj = e_ac_pj
        self.e_mac_fp32_pj = e_mac_fp32_pj
        self.e_mac_int8_pj = e_mac_int8_pj
        self.e_enc_pj = e_enc_pj
        self.e_leak_pj = e_leak_pj

    def profile_snn(self, model, spike_encoder, x_sample, time_steps=10):
        """
        Profiles a single transaction inference through the SNN including encoder overhead.
        Returns: SynOps, Total Energy (nJ), Latency (us), Sparsity (%)
        """
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)
            spikes = spike_encoder.encode(x_tensor)
            
            # Input Spike Sparsity
            total_elements = spikes.numel()
            active_spikes = spikes.sum().item()
            input_sparsity = 1.0 - (active_spikes / (total_elements + 1e-8))

            # Run inference and count SynOps
            out, synops = model(spikes)

            # End-to-end Energy Breakdown:
            # 1. Encoding Energy = Features * Timesteps * E_ENC
            num_features = x_tensor.shape[-1]
            encoding_pj = num_features * time_steps * self.e_enc_pj

            # 2. Dynamic Synaptic Accumulation Energy = SynOps * E_AC
            synops_pj = synops * self.e_ac_pj

            # 3. Static Leakage Energy = Neurons * Timesteps * E_LEAK
            num_hidden = getattr(model, 'hidden_dim', 64)
            leakage_pj = (num_hidden * 2 + 2) * time_steps * self.e_leak_pj

            total_energy_pj = encoding_pj + synops_pj + leakage_pj
            total_energy_nj = total_energy_pj / 1000.0

            # Modeled Embedded Latency on ARM Cortex-R52 (400 MHz) with TCM:
            cycles = (num_features * 2) + (synops * 1.2) + 150
            latency_us = cycles / 400.0

        return {
            "synops": synops,
            "energy_nj": total_energy_nj,
            "latency_us": latency_us,
            "input_sparsity_pct": input_sparsity * 100.0
        }

    def profile_int8_ann(self, input_dim=8, hidden_dim=64, num_classes=2):
        """
        Profiles INT8 Quantized MLP baseline for fair embedded TinyML comparison.
        """
        macs = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + (hidden_dim * num_classes)
        energy_pj = macs * self.e_mac_int8_pj + (hidden_dim * 2 * 0.05)
        energy_nj = energy_pj / 1000.0
        cycles = macs * 1.5 + 100  # INT8 SIMD instructions on Cortex-R52 / NEON
        latency_us = cycles / 400.0

        return {
            "macs": macs,
            "energy_nj": energy_nj,
            "latency_us": latency_us,
            "sparsity_pct": 0.0
        }

    def profile_dense_ann(self, input_dim=8, hidden_dim=64, num_classes=2):
        """
        Profiles standard unquantized FP32 dense ANN for comparison.
        """
        macs = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + (hidden_dim * num_classes)
        energy_pj = macs * self.e_mac_fp32_pj
        energy_nj = energy_pj / 1000.0
        cycles = macs * 8.0 + 300
        latency_us = cycles / 400.0

        return {
            "macs": macs,
            "energy_nj": energy_nj,
            "latency_us": latency_us,
            "sparsity_pct": 0.0
        }
