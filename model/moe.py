"""Batched mixture-of-experts built on ``mx.gather_mm``.

Self-contained (no ``mlx_lm`` dependency). This is the standard MLX pattern for
sparse MoE: expert weights are stored stacked as ``[num_experts, out, in]`` and
only the experts selected per token are evaluated, in one fused ``gather_mm``
call instead of a Python loop over experts.
"""

import mlx.core as mx
import mlx.nn as nn


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class SwitchLinear(nn.Module):
    """A stack of ``num_experts`` linear layers, indexed per token."""

    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = False
    ):
        super().__init__()
        scale = (1.0 / input_dims) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(num_experts, output_dims, input_dims)
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchGLU(nn.Module):
    """Batched SwiGLU experts: ``down(silu(gate(x)) * up(x))`` over selected experts."""

    def __init__(
        self, input_dims: int, hidden_dims: int, num_experts: int, bias: bool = False
    ):
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))

        # Sorting keeps each expert's tokens contiguous, which makes the gather
        # matmuls hit memory in order. Only worth it once there are many tokens.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(nn.silu(x_gate) * x_up, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
