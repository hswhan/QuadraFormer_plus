import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # 参数形状 [C, 1, 1]，自动匹配设备
        self.affine_weight = nn.Parameter(torch.ones(self.num_features, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features, 1, 1))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :]  # [B, 1, C]
        else:
            self.mean = x.mean(dim=dim2reduce, keepdim=True)

        var = x.var(dim=dim2reduce, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()
        self.stdev = torch.clamp(self.stdev, min=1e-3)  # ✅ 不让除以0

    def _normalize(self, x):
        if not x.is_contiguous():
            x = x.contiguous()

        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / (self.stdev + self.eps)  # ✅ 非 inplace，强制加 eps

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if not x.is_contiguous():
            x = x.contiguous()

        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)  # ✅ 改成 + eps

        x = x * (self.stdev + self.eps)  # ✅ 改成 + eps

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x