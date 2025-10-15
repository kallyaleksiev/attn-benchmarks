"""Torch.compile attention implementation."""

from typing import override

import torch
import torch.nn.functional as F

from ..exp_config import ExperimentResult, ExperimentRunner, ExperimentSpec


def expand_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, num_q_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand KV heads to match Q heads for GQA."""
    num_kv_heads = k.shape[1] if k.ndim == 4 else k.shape[2]
    if num_q_heads == num_kv_heads:
        return k, v

    num_groups = num_q_heads // num_kv_heads

    if k.ndim == 4:
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)
    else:
        k = k.repeat_interleave(num_groups, dim=2)
        v = v.repeat_interleave(num_groups, dim=2)

    return k, v


def _attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    k_expanded, v_expanded = expand_kv_for_gqa(k, v, q.shape[1])

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_expanded)

    return output


_compiled_attention = torch.compile(_attention_impl, mode="max-autotune")


class TorchCompileRunner(ExperimentRunner):
    def __init__(self):
        self.framework = "torch_compile"

    @override
    def preprocess(
        self,
        exp_spec: ExperimentSpec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = "cuda"
        dtype = exp_spec.dtype
        batch_size = exp_spec.batch_size
        q_len = exp_spec.q_seq_len
        kv_len = exp_spec.kv_seq_len
        num_q_heads = exp_spec.num_q_heads
        num_kv_heads = exp_spec.num_kv_heads
        head_dim = exp_spec.head_dim

        if exp_spec.workload_type == "decode":
            q = torch.randn(batch_size, num_q_heads, 1, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)
        else:
            q = torch.randn(batch_size, num_q_heads, q_len, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, dtype=dtype, device=device)

        return q, k, v

    @override
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        output = _compiled_attention(q, k, v)
        return output

    @override
    def run(
        self,
        exp_spec: ExperimentSpec,
        num_warmup: int = 3,
        num_iters: int = 20,
    ) -> ExperimentResult:
        preprocess_start = torch.cuda.Event(enable_timing=True)
        preprocess_end = torch.cuda.Event(enable_timing=True)

        preprocess_start.record()
        q, k, v = self.preprocess(exp_spec)
        preprocess_end.record()
        torch.cuda.synchronize()

        preprocess_time_ms = preprocess_start.elapsed_time(preprocess_end)

        for _ in range(num_warmup):
            _ = self.compute_attention(q, k, v)
            torch.cuda.synchronize()

        kernel_times = []
        for _ in range(num_iters):
            kernel_start = torch.cuda.Event(enable_timing=True)
            kernel_end = torch.cuda.Event(enable_timing=True)

            kernel_start.record()
            output = self.compute_attention(q, k, v)
            kernel_end.record()
            torch.cuda.synchronize()

            kernel_time_ms = kernel_start.elapsed_time(kernel_end)
            kernel_times.append(kernel_time_ms)

        import statistics

        kernel_time_ms = statistics.median(kernel_times)

        return ExperimentResult(
            exp_spec=exp_spec,
            framework=self.framework,
            o=output,
            preprocess_time=int(preprocess_time_ms * 1_000_000),
            kernel_time=int(kernel_time_ms * 1_000_000),
        )
