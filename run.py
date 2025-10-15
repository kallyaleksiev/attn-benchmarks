import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import click
import torch

from benchmarks.exp_config import (
    LLAMA_3_1_8B,
    QWEN3_30B_A3B,
    QWEN3_235B_A22B,
    ExperimentSpec,
    get_all_exp_configs,
)
from benchmarks.frameworks.flash_attn import FlashAttnRunner
from benchmarks.frameworks.flashinfer import FlashInferRunner
from benchmarks.frameworks.flex_attention import FlexAttentionRunner
from benchmarks.frameworks.naive_torch import NaiveTorchRunner
from benchmarks.frameworks.pytorch_sdpa import PytorchSDPARunner
from benchmarks.frameworks.te_attn import TEAttnRunner
from benchmarks.frameworks.torch_compile import TorchCompileRunner

FRAMEWORKS = {
    "flashinfer": FlashInferRunner,
    "flash_attn": FlashAttnRunner,
    "te_attn": TEAttnRunner,
    "pytorch_sdpa": PytorchSDPARunner,
    "torch_compile": TorchCompileRunner,
    "naive_torch": NaiveTorchRunner,
    "flex_attention": FlexAttentionRunner,
}

MODELS = {
    "llama-3.1-8b": LLAMA_3_1_8B,
    "qwen3-30b-a3b": QWEN3_30B_A3B,
    "qwen3-235b-a22b": QWEN3_235B_A22B,
}


@dataclass
class BenchmarkResult:
    model_name: str
    workload_type: str
    framework: str
    batch_size: int
    q_seq_len: int
    kv_seq_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    preprocess_time_ns: int
    kernel_time_ns: int
    kernel_time_ms: float
    throughput_tokens_per_sec: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    error: Optional[str] = None


def should_skip(
    framework: str, exp_spec: ExperimentSpec, available_memory_gb: float
) -> tuple[bool, str]:
    bytes_per_element = 2
    q_size = exp_spec.batch_size * exp_spec.num_q_heads * exp_spec.q_seq_len * exp_spec.head_dim
    k_size = exp_spec.batch_size * exp_spec.num_kv_heads * exp_spec.kv_seq_len * exp_spec.head_dim
    v_size = exp_spec.batch_size * exp_spec.num_kv_heads * exp_spec.kv_seq_len * exp_spec.head_dim
    scores_size = (
        exp_spec.batch_size * exp_spec.num_q_heads * exp_spec.q_seq_len * exp_spec.kv_seq_len
    )
    output_size = (
        exp_spec.batch_size * exp_spec.num_q_heads * exp_spec.q_seq_len * exp_spec.head_dim
    )

    total_bytes = (q_size + k_size + v_size + output_size) * bytes_per_element + scores_size * 4
    total_bytes *= 1.5
    estimated_gb = total_bytes / (1024**3)

    if estimated_gb > available_memory_gb:
        return True, f"Estimated memory {estimated_gb:.1f}GB exceeds available"

    return False, ""


def run_benchmark(
    framework_name: str,
    exp_spec: ExperimentSpec,
    num_warmup: int = 3,
    num_iters: int = 20,
) -> BenchmarkResult:
    try:
        runner_class = FRAMEWORKS[framework_name]
        runner = runner_class()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        result = runner.run(exp_spec, num_warmup=num_warmup, num_iters=num_iters)

        kernel_time_ms = result.kernel_time / 1_000_000
        total_tokens = exp_spec.batch_size * exp_spec.q_seq_len
        throughput = (total_tokens / kernel_time_ms) * 1000

        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)

        model_name = (
            f"H_q={exp_spec.num_q_heads},H_kv={exp_spec.num_kv_heads},D={exp_spec.head_dim}"
        )

        return BenchmarkResult(
            model_name=model_name,
            workload_type=exp_spec.workload_type,
            framework=framework_name,
            batch_size=exp_spec.batch_size,
            q_seq_len=exp_spec.q_seq_len,
            kv_seq_len=exp_spec.kv_seq_len,
            num_q_heads=exp_spec.num_q_heads,
            num_kv_heads=exp_spec.num_kv_heads,
            head_dim=exp_spec.head_dim,
            preprocess_time_ns=result.preprocess_time,
            kernel_time_ns=result.kernel_time,
            kernel_time_ms=kernel_time_ms,
            throughput_tokens_per_sec=throughput,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            error=None,
        )

    except Exception as e:
        model_name = (
            f"H_q={exp_spec.num_q_heads},H_kv={exp_spec.num_kv_heads},D={exp_spec.head_dim}"
        )
        return BenchmarkResult(
            model_name=model_name,
            workload_type=exp_spec.workload_type,
            framework=framework_name,
            batch_size=exp_spec.batch_size,
            q_seq_len=exp_spec.q_seq_len,
            kv_seq_len=exp_spec.kv_seq_len,
            num_q_heads=exp_spec.num_q_heads,
            num_kv_heads=exp_spec.num_kv_heads,
            head_dim=exp_spec.head_dim,
            preprocess_time_ns=0,
            kernel_time_ns=0,
            kernel_time_ms=0.0,
            throughput_tokens_per_sec=0.0,
            memory_allocated_gb=0.0,
            memory_reserved_gb=0.0,
            error=str(e),
        )
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_results(results: list[BenchmarkResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to: {json_path}")


@click.command()
@click.option("--frameworks", multiple=True, default=list(FRAMEWORKS.keys()))
@click.option("--models", multiple=True, default=list(MODELS.keys()))
@click.option("--workload-types", multiple=True, default=["prefill", "decode", "append"])
@click.option("--output-dir", default="./results", type=click.Path())
@click.option("--num-warmup", type=int, default=3)
@click.option("--num-iters", type=int, default=20)
def main(frameworks, models, workload_types, output_dir, num_warmup, num_iters):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    all_configs = get_all_exp_configs()

    selected_models = [MODELS[m] for m in models]
    configs = [
        c
        for c in all_configs
        if c.workload_type in workload_types
        and any(
            c.num_q_heads == m.num_attention_heads
            and c.num_kv_heads == m.num_key_value_heads
            and c.head_dim == m.head_dim
            for m in selected_models
        )
    ]

    total = len(configs) * len(frameworks)
    print(f"Running {total} benchmarks\n")

    results = []
    current = 0

    for exp_spec in configs:
        for framework_name in frameworks:
            current += 1

            skip, reason = should_skip(framework_name, exp_spec, available_memory_gb)
            if skip:
                print(
                    f"[{current}/{total}] SKIP {framework_name} | {exp_spec.workload_type} | B={exp_spec.batch_size} Q={exp_spec.q_seq_len} KV={exp_spec.kv_seq_len} | {reason}"
                )
                continue

            print(
                f"[{current}/{total}] {framework_name} | {exp_spec.workload_type} | B={exp_spec.batch_size} Q={exp_spec.q_seq_len} KV={exp_spec.kv_seq_len}"
            )

            result = run_benchmark(framework_name, exp_spec, num_warmup, num_iters)

            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(
                    f"  Kernel: {result.kernel_time_ms:.2f} ms | Throughput: {result.throughput_tokens_per_sec:.0f} tok/s | Memory: {result.memory_allocated_gb:.2f} GB"
                )

            results.append(result)

    save_results(results, Path(output_dir))

    successful = sum(1 for r in results if r.error is None)
    errors = sum(1 for r in results if r.error is not None)

    print(f"\n{'=' * 80}")
    print(f"Complete! Total: {len(results)} | Success: {successful} | Errors: {errors}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
