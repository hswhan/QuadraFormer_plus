import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableDilationExpert(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 d_min: int = 1, d_max: int = 8, use_dilated_conv: bool = False):
        super().__init__()
        assert in_channels == out_channels
        assert d_min >= 1 and d_max >= d_min
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.use_dilated_conv = bool(use_dilated_conv)

        self.d_logit = nn.Parameter(torch.empty(1).uniform_(-2.0, 2.0))

        pad = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size,
                            padding=pad, groups=in_channels, bias=True)

        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

        self.gn = nn.GroupNorm(num_groups=1, num_channels=out_channels)

        self.gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3)≈0.047

    @torch.no_grad()
    def current_dilation(self) -> int:
        d_cont = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(self.d_logit)
        d_int = torch.clamp(torch.round(d_cont), self.d_min, self.d_max)
        return int(d_int.item())

    def _ste_round(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.round(x)
        return x + (y - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        return: [B, C, L’]
        """
        assert x.dim() == 3, f"Expect [B,C,L], got {x.shape}"
        B, C, L = x.shape

        d_cont = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(self.d_logit)
        d_int = self._ste_round(d_cont)
        d_int = torch.clamp(d_int, self.d_min, self.d_max).item()
        d_int = int(d_int)

        if d_int > 1:
            x_ds = F.avg_pool1d(x, kernel_size=d_int, stride=d_int, ceil_mode=False)
        else:
            x_ds = x

        y = self.dw(x_ds)
        y = F.gelu(y)
        y = self.pw(y)
        y = self.gn(y)

        g = torch.sigmoid(self.gate)
        out = x_ds + g * y
        return out
