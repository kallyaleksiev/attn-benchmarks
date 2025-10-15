"""Flash Attention implementation https://github.com/Dao-AILab/flash-attention"""

from typing import override

import torch
from flash_attn import flash_attn_func

from ..exp_config import ExperimentResult, ExperimentRunner, ExperimentSpec


class FlashAttnRunner(ExperimentRunner):
    def __init__(self):
        self.framework = "flash_attn"

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
            q = torch.randn(batch_size, 1, num_q_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        else:
            q = torch.randn(batch_size, q_len, num_q_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        return q, k, v

    @override
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False, softmax_scale=None)
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
