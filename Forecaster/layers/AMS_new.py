# layers/AMS_new.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Forecaster.layers.adaptive_patch import LearnableDilationExpert
from Forecaster.layers.QuadraLayer import QuadraLayer
from Forecaster.util.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP


class AMS_new(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 num_experts,
                 device,
                 num_nodes=1,
                 d_model=32,
                 d_ff=64,
                 dynamic=False,
                 patch_size=[8, 6, 4, 12],
                 noisy_gating=True,
                 k=4,
                 layer_number=1,
                 residual_connection=1,
                 batch_norm=False,
                 adaptive_d_min: int = 1,
                 adaptive_d_max: int = 1,
                 adaptive_kernel_size: int = 3,
                 adaptive_use_dilated_conv: bool = True,
                 lambda_diverse: float = 1e-3,
                 eps_var: float = 1e-4,
                 extractor_tau: float = 0.0,
                 debug=False,
                 adaptive_patch: bool = True):
        super(AMS_new, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k
        self.device = device
        self.debug = debug

        self.use_adaptive_patch = bool(adaptive_patch)
        self.lambda_diverse = float(lambda_diverse if self.use_adaptive_patch else 0.0)
        self.eps_var = float(eps_var)
        self.extractor_tau = float(extractor_tau)
        self.bypass_extractors = (not self.use_adaptive_patch)

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(
                QuadraLayer(device=device, d_model=d_model, d_ff=d_ff,
                            dynamic=dynamic, num_nodes=num_nodes, patch_nums=patch_nums,
                            patch_size=patch, factorized=True, layer_number=layer_number, batch_norm=batch_norm)
            )

        # gating networks
        self.w_noise = nn.Linear(input_size, num_experts)
        self.w_gate = nn.Linear(input_size, num_experts)

        self.residual_connection = residual_connection
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert (self.k <= self.num_experts)

        if self.use_adaptive_patch:
            self.extractors = nn.ModuleList([
                LearnableDilationExpert(
                    in_channels=num_nodes,
                    out_channels=num_nodes,
                    kernel_size=adaptive_kernel_size,
                    d_min=adaptive_d_min,
                    d_max=adaptive_d_max,
                    use_dilated_conv=adaptive_use_dilated_conv,
                ) for _ in range(self.num_experts)
            ])
        else:
            self.extractors = nn.ModuleList()

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var(dim=0) / (x.float().mean(dim=0) ** 2 + eps)

    def _gates_to_load(self, gates):
        # count how many samples selected each expert (per column)
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        Robust computation of P(in top-k) using a numerically stable Normal CDF implementation via erf.
        clean_values, noisy_values: (batch, num_experts)
        noisy_top_values: top-k+1 values per row (shape (batch, k+1) or similar)
        """
        eps = 1e-6
        noise_stddev = torch.clamp(noise_stddev, min=eps)

        clean_values = torch.nan_to_num(clean_values, nan=0.0, posinf=1e6, neginf=-1e6)
        noisy_values = torch.nan_to_num(noisy_values, nan=0.0, posinf=1e6, neginf=-1e6)
        noisy_top_values = torch.nan_to_num(noisy_top_values, nan=0.0, posinf=1e6, neginf=-1e6)

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)

        top_values_flat = noisy_top_values.flatten()
        idx_k = torch.arange(batch, device=clean_values.device) * m + min(self.k, m - 1)
        idx_k_minus1 = torch.clamp(idx_k - 1, min=0)

        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, idx_k), 1)
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, idx_k_minus1), 1)

        z_in = (clean_values - threshold_if_in) / noise_stddev
        z_out = (clean_values - threshold_if_out) / noise_stddev
        z_in = torch.clamp(z_in, min=-50.0, max=50.0)
        z_out = torch.clamp(z_out, min=-50.0, max=50.0)

        sqrt2 = math.sqrt(2.0)
        prob_if_in = 0.5 * (1.0 + torch.erf(z_in / sqrt2))
        prob_if_out = 0.5 * (1.0 + torch.erf(z_out / sqrt2))

        prob_if_in = torch.clamp(torch.nan_to_num(prob_if_in, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        prob_if_out = torch.clamp(torch.nan_to_num(prob_if_out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        is_in = torch.gt(noisy_values, threshold_if_in)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        prob = torch.clamp(torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return prob

    def seasonality_and_trend_decompose(self, x):
        x_orig = x
        try:
            x2 = x_orig[:, :, :, 0]
        except Exception:
            x2 = x_orig
        _, trend = self.trend_model(x2)
        seasonality, _ = self.seasonality_model(x2)
        return x2 + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        x: input tensor, expected shape (..., num_nodes) matching start_linear
        returns: gates (batch, num_experts), load (num_experts)
        """
        x_lin = self.start_linear(x).squeeze(-1)
        x_lin = torch.nan_to_num(x_lin, nan=0.0, posinf=1e6, neginf=-1e6)

        clean_logits = self.w_gate(x_lin)
        bl_loss = None
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x_lin)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise_stddev = torch.clamp(noise_stddev, min=1e-3, max=1e3)
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
            noisy_logits = torch.nan_to_num(noisy_logits, nan=0.0, posinf=1e3, neginf=-1e3)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        k_plus = min(self.k + 1, self.num_experts)
        top_logits, top_indices = logits.topk(k_plus, dim=1)
        top_k_logits = top_logits[:, :self.k] if self.k <= top_logits.size(1) else top_logits
        top_k_indices = top_indices[:, :self.k] if self.k <= top_indices.size(1) else top_indices

        try:
            top_k_gates = self.softmax(top_k_logits)
        except Exception:
            top_k_gates = F.softmax(top_k_logits, dim=1)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            try:
                prob = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)
            except Exception as e:
                if self.debug:
                    print(f"[AMS_new] _prob_in_top_k exception: {e}. Sanitizing and retrying.")
                clean_logits = torch.nan_to_num(clean_logits, nan=0.0, posinf=1e6, neginf=-1e6)
                noisy_logits = torch.nan_to_num(noisy_logits, nan=0.0, posinf=1e6, neginf=-1e6)
                noise_stddev = torch.clamp(noise_stddev, min=1e-6, max=1e3)
                top_logits = torch.nan_to_num(top_logits, nan=0.0, posinf=1e6, neginf=-1e6)
                prob = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)

            prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
            prob = torch.clamp(prob, 0.0, 1.0)
            load = prob.sum(0)
        else:
            load = self._gates_to_load(gates)

        gates = torch.nan_to_num(gates, nan=0.0, posinf=1e6, neginf=0.0)
        gates = torch.clamp(gates, min=0.0, max=1.0)
        load = torch.nan_to_num(load, nan=0.0, posinf=1e6, neginf=0.0)

        if self.debug:
            print(f"[AMS_new] noisy_top_k_gating: gates min/max {gates.min().item()}/{gates.max().item()}, load min/max {load.min().item()}/{load.max().item()}")
        return gates, load

    def forward(self, x, loss_coef=1e-4):

        new_x = self.seasonality_and_trend_decompose(x)

        # gating
        gates, load = self.noisy_top_k_gating(new_x, self.training)

        # balance loss
        importance = gates.sum(0)
        balance_loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)  # list of [b_i, T, C, E]

        T_target = x.size(1)
        E_target = x.size(3) if x.dim() == 4 else 1

        expert_outputs = []
        for i in range(self.num_experts):
            inp = expert_inputs[i]  # [b_i, T, C, E]
            if (inp is None) or (inp.numel() == 0):
                expert_outputs.append(inp)
                continue

            # [B,T,C,E] -> [B,C,T,E] -> [B,C,L=T*E]
            inp_c = inp.permute(0, 2, 1, 3).contiguous()
            B_i, C_i, T_i, E_i = inp_c.shape
            inp_c = inp_c.view(B_i, C_i, T_i * E_i)

            if self.bypass_extractors or (len(self.extractors) == 0):
                out_c = inp_c
            else:
                out_c = self.extractors[i](inp_c)
                L_target = T_target * E_target
                out_c = F.interpolate(out_c, size=L_target, mode='linear', align_corners=False)

            tau = getattr(self, "extractor_tau", 0.1)
            out_c = tau * out_c + (1.0 - tau) * inp_c

            out_c = out_c.view(B_i, C_i, T_target, E_target).contiguous()
            out_q = out_c.permute(0, 2, 1, 3).contiguous()  # [B, T, C, E]

            y, _ = self.experts[i](out_q)
            expert_outputs.append(y)

        output = dispatcher.combine(expert_outputs)

        diverse_loss = 0.0
        if self.use_adaptive_patch and self.lambda_diverse > 0.0 and len(self.extractors) > 0:
            with torch.no_grad():
                ds = [ex.current_dilation() for ex in self.extractors]  # list[int]
            d_tensor = torch.tensor(ds, dtype=torch.float32, device=output.device)
            var = torch.var(d_tensor, unbiased=False)
            diverse_loss = 1.0 / (var + self.eps_var)

        balance_loss = balance_loss + self.lambda_diverse * diverse_loss
        return output, balance_loss
