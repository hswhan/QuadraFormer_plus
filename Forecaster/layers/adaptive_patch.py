import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDilationExpert(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        d_min: int = 1,
        d_max: int = 16,
        use_dilated_conv: bool = False,
        temperature: float = 1.0,
        hard_select: bool = True,
    ):
        super().__init__()

        assert in_channels == out_channels,  "require in_channels == out_channels"
        assert d_min >= 1 and d_max >= d_min

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.use_dilated_conv = bool(use_dilated_conv)
        self.temperature = float(temperature)
        self.hard_select = bool(hard_select)

        candidates = torch.arange(self.d_min, self.d_max + 1, dtype=torch.long)
        self.register_buffer("d_candidates", candidates)
        num_candidates = len(candidates)

        self.scale_logits = nn.Parameter(0.01 * torch.randn(num_candidates))

        pad = (kernel_size - 1) // 2

        self.dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=pad,
            groups=in_channels,
            bias=True,
        )

        self.pw = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

        self.gn = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.gate = nn.Parameter(torch.tensor(-3.0))

    def current_dilation(self) -> int:

        with torch.no_grad():
            idx = torch.argmax(self.scale_logits).item()
            return int(self.d_candidates[idx].item())

    def scale_probabilities(self) -> torch.Tensor:

        return F.softmax(self.scale_logits / self.temperature, dim=0)

    def _ste_one_hot(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Straight-through categorical estimator.

        forward: hard one-hot
        backward: soft probabilities
        """
        if not self.hard_select:
            return probs

        idx = torch.argmax(probs)
        hard = torch.zeros_like(probs)
        hard[idx] = 1.0

        # forward = hard, backward = probs
        alpha = hard + probs - probs.detach()
        return alpha

    def _pool_branch(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """
        AvgPool_d -> DWConv -> PWConv -> GN -> gated residual -> Upsample to L.
        x: [B, C, L]
        return: [B, C, L]
        """
        B, C, L = x.shape

        if d > 1:
            x_ds = F.avg_pool1d(
                x,
                kernel_size=d,
                stride=d,
                ceil_mode=True,
            )
        else:
            x_ds = x

        y = self.dw(x_ds)
        y = F.gelu(y)
        y = self.pw(y)
        y = self.gn(y)

        out = x_ds + torch.sigmoid(self.gate) * y


        if out.size(-1) != L:
            out = F.interpolate(
                out,
                size=L,
                mode="linear",
                align_corners=False,
            )

        return out

    def _dilated_conv_branch(self, x: torch.Tensor, d: int) -> torch.Tensor:

        pad = ((self.kernel_size - 1) // 2) * d

        y = F.conv1d(
            x,
            weight=self.dw.weight,
            bias=self.dw.bias,
            stride=1,
            padding=pad,
            dilation=d,
            groups=self.in_channels,
        )


        if y.size(-1) > x.size(-1):
            y = y[..., :x.size(-1)]
        elif y.size(-1) < x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)

        y = F.gelu(y)
        y = self.pw(y)
        y = self.gn(y)

        out = x + torch.sigmoid(self.gate) * y
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        return: [B, C, L]
        """
        assert x.dim() == 3, f"Expect [B, C, L], got {x.shape}"

        probs = self.scale_probabilities()
        alpha = self._ste_one_hot(probs)

        branch_outputs = []

        for d_tensor in self.d_candidates:
            d = int(d_tensor.item())

            if self.use_dilated_conv:
                y_d = self._dilated_conv_branch(x, d)
            else:
                y_d = self._pool_branch(x, d)

            branch_outputs.append(y_d)

        # [M, B, C, L]
        stacked = torch.stack(branch_outputs, dim=0)

        # [M, 1, 1, 1]
        alpha = alpha.view(-1, 1, 1, 1)

        # forward hard-select one branch; backward through softmax probabilities
        out = torch.sum(alpha * stacked, dim=0)

        return out