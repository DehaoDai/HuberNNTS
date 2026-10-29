"""Bounded two-hidden-layer ReLU network and clipped sparsity penalty."""

import torch
from torch import nn


class ClippedNetwork(nn.Module):
    def __init__(self, d, width, depth, output_bound, parameter_bound, lip_bound):
        super().__init__()
        if depth < 1 or min(output_bound, parameter_bound, lip_bound) <= 0:
            raise ValueError("Depth and network bounds must be positive.")
        sizes = [d] + [width] * depth + [1]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.output_bound = output_bound
        self.parameter_bound = parameter_bound
        self.layer_row_bound = lip_bound ** (1.0 / len(self.layers))
        self.project_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return torch.clamp(self.layers[-1](x), -self.output_bound, self.output_bound)

    @torch.no_grad()
    def project_parameters(self):
        """Enforce |theta|_infinity <= B and Lip_infinity(f) <= Lambda."""
        for layer in self.layers:
            layer.weight.clamp_(-self.parameter_bound, self.parameter_bound)
            layer.bias.clamp_(-self.parameter_bound, self.parameter_bound)
            row_norm = layer.weight.abs().sum(dim=1, keepdim=True)
            scale = (self.layer_row_bound / row_norm.clamp_min(1e-12)).clamp(max=1.0)
            layer.weight.mul_(scale)

    def clipped_penalty(self, omega):
        if omega <= 0:
            raise ValueError("omega must be positive.")
        return sum((p.abs() / omega).clamp(max=1.0).sum() for p in self.parameters())

    def active_count(self, omega):
        return sum(int((p.abs() >= omega).sum()) for p in self.parameters())


