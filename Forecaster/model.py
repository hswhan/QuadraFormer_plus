import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS_new import AMS_new
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from functools import reduce
from operator import mul
from statsmodels.tsa.arima.model import ARIMA
import os, subprocess
import numpy as np
import torch
import torch.nn as nn
# model.py


class QuadraFormer(nn.Module):
    def __init__(self, **kwargs):
        params = {**kwargs}
        super(QuadraFormer, self).__init__()
        self.layer_nums = 1
        # self.layer_nums = params["num_layers"]  # --layer_nums 3
        self.num_nodes = params["input_dim"]
        self.pre_len = params["prediction_length"]
        self.seq_len = params["window_size"]
        self.k = 2  # --k 2
        self.num_experts_list = [4]
        self.patch_size_list = [[16, 8, 4, 2]]
        self.d_model = 16  # --d_model 16
        self.d_ff = 64  # --d_ff 64
        self.residual_connection = True  # --residual_connection 0
        self.batch_norm = False  # --batch_norm 0
        self.revin = True  # --revin 1
        self.drop = 0.1  # --drop 0.1

        if self.revin:
            self.revin_layer = RevIN(
                num_features=self.num_nodes,
                affine=False,
                subtract_last=False
            )

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:0')

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS_new(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number= num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        balance_loss = torch.tensor(0.0, device=x.device)
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))
        batch_size = x.shape[0]
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss.to(x.device)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        res = out
        out = self.projections(out).transpose(2, 1)
        out += res.mean(dim=2).unsqueeze(1)
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        return out, balance_loss


class QuadraFormer_star(nn.Module):
    default_model_params = {
        'input_dim': 209,
        'output_dim': 209,
        'num_layers': 3,
        'prediction_length': 16,
        'window_size': 16,
    }
    def __init__(self, **kwargs):
        params = {**self.default_model_params, **kwargs}
        super(QuadraFormer_star, self).__init__()
        self.layer_nums = 1  # --layer_nums 3
        # self.layer_nums = params["num_layers"]  # --layer_nums 3
        self.num_nodes = params["input_dim"]
        self.pre_len = params["prediction_length"]  # --pred_len 96
        self.seq_len = params["window_size"]  # --seq_len 96
        self.k = 2  # --k 2
        self.num_experts_list = [4]  # --num_experts_list [4,4,4]
        self.patch_size_list = [[16, 8, 4, 2]]

        self.d_model = 16  # --d_model 16
        self.d_ff = 64  # --d_ff 64

        self.residual_connection = True  # --residual_connection 0
        self.batch_norm = False  # --batch_norm 0
        self.revin = True  # --revin 1
        self.drop = 0.1  # --drop 0.1

        if self.revin:
            self.revin_layer = RevIN(
                num_features=self.num_nodes,  
                affine=False,  
                subtract_last=False
            )

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)

        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:0')  

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS_new(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number= num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        # self.projections = nn.Sequential(
        #     nn.Conv1d(in_channels=self.seq_len, out_channels=self.pre_len, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 1)
        # )

        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        balance_loss = torch.tensor(0.0, device=x.device)

        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # print("x stats:", x.min().item(), x.max().item(), x.mean().item(), x.std().item())
        out = self.start_fc(x.unsqueeze(-1))
        # print("out stats:", out.min().item(), out.max().item(), out.mean().item(), out.std().item())
        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss.to(x.device)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        res = out
        out = self.projections(out).transpose(2, 1)
        out += res.mean(dim=2).unsqueeze(1)  # [B, 1, D] → broadcast

        # out = out.permute(0, 2, 1, 3).reshape(batch_size * self.num_nodes, self.seq_len, self.d_model)
        # out = self.projections(out).squeeze(-1)  # [B*N, pre_len]
        # out = out.view(batch_size, self.num_nodes, self.pre_len).transpose(1, 2)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss


import torch
import torch.nn as nn
from Forecaster.layers.QuadraLayer import QuadraLayer
from Forecaster.layers.AMS_new import AMS_new
from layers.RevIN import RevIN

# No-op cross attention module
class NoOpCrossAttention(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

# QuadraLayer without cross-attention
class QuadraLayer_woc(QuadraLayer):
    def __init__(self, device, d_model, d_ff, num_nodes, patch_nums, patch_size,
                 factorized, layer_number, batch_norm):
        super().__init__(device=device,
                         d_model=d_model,
                         d_ff=d_ff,
                         num_nodes=num_nodes,
                         patch_nums=patch_nums,
                         patch_size=patch_size,
                         dynamic=False,
                         factorized=factorized,
                         layer_number=layer_number,
                         batch_norm=batch_norm)
        # Override cross-attention with no-op
        self.intra_cross_attention = NoOpCrossAttention()
        self.inter_cross_attention = NoOpCrossAttention()



# AMS router block without cross-attention
class AMS_woc(AMS_new):
    def __init__(self, input_size, output_size, num_experts, device,
                 num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8,6,4,2], noisy_gating=True, k=4,
                 layer_number=1, residual_connection=1, batch_norm=False):
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         num_experts=num_experts,
                         device=device,
                         num_nodes=num_nodes,
                         d_model=d_model,
                         d_ff=d_ff,
                         dynamic=dynamic,
                         patch_size=patch_size,
                         noisy_gating=noisy_gating,
                         k=k,
                         layer_number=layer_number,
                         residual_connection=residual_connection,
                         batch_norm=batch_norm)
        # Replace experts with cross-attention-ablated version
        self.experts = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(
                QuadraLayer_woc(device=device,
                                d_model=d_model,
                                d_ff=d_ff,
                                num_nodes=num_nodes,
                                patch_nums=patch_nums,
                                patch_size=patch,
                                factorized=True,
                                layer_number=layer_number,
                                batch_norm=batch_norm)
            )

# QuadraFormer without cross-attention
class QuadraFormer_woc(nn.Module):
    default_model_params = {
        'input_dim': 209,
        'output_dim': 209,
        'num_layers': 3,
        'prediction_length': 16,
        'window_size': 16,
    }
    def __init__(self, **kwargs):
        params = {**self.default_model_params, **kwargs}
        super(QuadraFormer_woc, self).__init__()
        self.layer_nums = 1
        self.num_nodes = params['input_dim']
        self.pre_len = params['prediction_length']
        self.seq_len = params['window_size']
        self.k = 2
        self.num_experts_list = [4]
        # Use same patch sizes as original QuadraFormer
        self.patch_size_list = [[16, 8, 4, 2]]

        # Model hyperparams
        self.d_model = 16
        self.d_ff = 64
        self.residual_connection = True
        self.batch_norm = False
        self.revin = True
        self.drop = 0.1

        # RevIN layer
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)

        # Input projection
        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)

        # AMS blocks
        self.AMS_lists = nn.ModuleList()
        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS_woc(input_size=self.seq_len,
                        output_size=self.seq_len,
                        num_experts=self.num_experts_list[num],
                        device=torch.device('cuda:0'),
                        num_nodes=self.num_nodes,
                        d_model=self.d_model,
                        d_ff=self.d_ff,
                        dynamic=False,
                        patch_size=self.patch_size_list[num],
                        noisy_gating=True,
                        k=self.k,
                        layer_number=num+1,
                        residual_connection=self.residual_connection,
                        batch_norm=self.batch_norm)
            )

        # Output projection
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        balance_loss = torch.tensor(0.0, device=x.device)
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))
        batch_size = x.shape[0]

        # Pass through AMS blocks
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss.to(x.device)

        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        res = out
        out = self.projections(out).transpose(2, 1)
        out += res.mean(dim=2).unsqueeze(1)

        if self.revin:
            out = self.revin_layer(out, 'denorm')
        return out, balance_loss


class QuadraFormer_woa(nn.Module):
    """
    QuadraFormer without Adaptive Multi-Scale selection (no AMS module).
    Only a single-scale QuadraLayer is applied.
    """
    default_model_params = {
        'input_dim': 209,
        'output_dim': 209,
        'prediction_length': 16,
        'window_size': 16,
    }

    def __init__(self, **kwargs):
        params = {**self.default_model_params, **kwargs}
        super(QuadraFormer_woa, self).__init__()

        # Model config
        self.num_nodes = params['input_dim']
        self.pre_len = params['prediction_length']
        self.seq_len = params['window_size']

        # Hyperparameters
        self.d_model = 16
        self.d_ff = 64
        self.batch_norm = False
        self.revin = True

        # RevIN normalization
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)

        # Input projection: from scalar to d_model
        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)

        # Single-scale QuadraLayer: patch_size = seq_len, so only one patch
        self.layer = QuadraLayer(
            device=torch.device('cuda:0'),
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_nodes=self.num_nodes,
            patch_nums=1,
            patch_size=self.seq_len,
            dynamic=False,
            factorized=True,
            layer_number=1,
            batch_norm=self.batch_norm
        )

        # Output projection
        self.projections = nn.Linear(self.seq_len * self.d_model, self.pre_len)

    def forward(self, x):
        # x: [B, seq_len, num_nodes]
        balance_loss = torch.tensor(0.0, device=x.device)

        # Normalize
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # Input embedding
        out = self.start_fc(x.unsqueeze(-1))  # [B, seq_len, num_nodes, d_model]
        batch_size = x.size(0)

        # Single QuadraLayer
        out, _ = self.layer(out)
        # out: [B, seq_len, num_nodes, d_model]

        # Reshape for projection
        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        res = out

        # Project to prediction length
        out = self.projections(out).transpose(2, 1)  # [B, pre_len, num_nodes]

        # Add mean-based residual
        out = out + res.mean(dim=2).unsqueeze(1)

        # Denormalize
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss



from Forecaster.layers.QuadraLayer import SparseScaledDotProductAttention, ScaledDotProductAttention

from layers.QuadraLayer import SparseScaledDotProductAttention, ScaledDotProductAttention

def _replace_sparse(module):
    for name, child in module.named_children():
        if isinstance(child, SparseScaledDotProductAttention):
            dense = ScaledDotProductAttention(
                d_model=child.d_model,
                n_heads=child.n_heads,
                attn_dropout=child.attn_dropout.p,  
                res_attention=child.res_attention,
                lsa=child.lsa
            )
            setattr(module, name, dense)
        else:
            _replace_sparse(child)


class QuadraFormer_wos(QuadraFormer):

    def __init__(self, **kwargs):
        super(QuadraFormer_wos, self).__init__(**kwargs)
        _replace_sparse(self)


class QuadraFormer_star_RE(nn.Module):
    default_model_params = {
        'input_dim': 209,
        'output_dim': 209,
        'num_layers': 3,
        'prediction_length': 16,
        'window_size': 16,
    }
    def __init__(self, **kwargs):
        params = {**self.default_model_params, **kwargs}
        super(QuadraFormer_star_RE, self).__init__()
        self.layer_nums = 1  # --layer_nums 3
        # self.layer_nums = params["num_layers"]  # --layer_nums 3
        self.num_nodes = params["input_dim"]
        self.pre_len = params["prediction_length"]  # --pred_len 96
        self.seq_len = params["window_size"]  # --seq_len 96
        self.k = 2  # --k 2
        self.num_experts_list = [4]  # --num_experts_list [4,4,4]
        self.patch_size_list = [[16, 8, 4, 2]] 

        self.d_model = 16  # --d_model 16
        self.d_ff = 64  # --d_ff 64

        self.residual_connection = True  # --residual_connection 0
        self.batch_norm = False  # --batch_norm 0
        self.revin = True  # --revin 1
        self.drop = 0.1  # --drop 0.1

        if self.revin:
            self.revin_layer = RevIN(
                num_features=self.num_nodes,  
                affine=False,  
                subtract_last=False
            )

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)

        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:0')  

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS_new(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number= num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        # self.projections = nn.Sequential(
        #     nn.Conv1d(in_channels=self.seq_len, out_channels=self.pre_len, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 1)
        # )

        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        balance_loss = torch.tensor(0.0, device=x.device)

        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # print("x stats:", x.min().item(), x.max().item(), x.mean().item(), x.std().item())
        out = self.start_fc(x.unsqueeze(-1))
        # print("out stats:", out.min().item(), out.max().item(), out.mean().item(), out.std().item())
        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss.to(x.device)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        res = out
        out = self.projections(out).transpose(2, 1)
        out += res.mean(dim=2).unsqueeze(1)  # [B, 1, D] → broadcast

        # out = out.permute(0, 2, 1, 3).reshape(batch_size * self.num_nodes, self.seq_len, self.d_model)
        # out = self.projections(out).squeeze(-1)  # [B*N, pre_len]
        # out = out.view(batch_size, self.num_nodes, self.pre_len).transpose(1, 2)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss

