"""Run benchmarks for README table with specific configurations."""

import json
from pathlib import Path

import torch

from benchmarks.exp_config import (
    LLAMA_3_1_8B,
    QWEN3_235B_A22B,
    QWEN3_30B_A3B,
    ExperimentSpec,
)
from benchmarks.frameworks.flash_attn import FlashAttnRunner
from benchmarks.frameworks.flashinfer import FlashInferRunner
from benchmarks.frameworks.flex_attention import FlexAttentionRunner
from benchmarks.frameworks.naive_torch import NaiveTorchRunner
from benchmarks.frameworks.pytorch_sdpa import PytorchSDPARunner
from benchmarks.frameworks.te_attn import TEAttnRunner
from benchmarks.frameworks.torch_compile import TorchCompileRunner

FRAMEWORKS = {
    "flash_attn": FlashAttnRunner,
    "flashinfer": FlashInferRunner,
    "te_attn": TEAttnRunner,
    "flex_attention": FlexAttentionRunner,
    "torch_compile": TorchCompileRunner,
    "naive_torch": NaiveTorchRunner,
    "pytorch_sdpa": PytorchSDPARunner,
}

MODELS = {
    "llama-3.1-8b": LLAMA_3_1_8B,
    "qwen3-30b-a3b": QWEN3_30B_A3B,
    "qwen3-235b-a22b": QWEN3_235B_A22B,
}

BATCH_SIZE = 256
SEQ_LEN = 4096
APPEND_Q_CHUNK = 128


def create_experiment_specs():
    """Create experiment specs for all models and workload types."""
    specs = []

    for model_name, model_config in MODELS.items():
        prefill_spec = ExperimentSpec(
            workload_type="prefill",
            batch_size=BATCH_SIZE,
            q_seq_len=SEQ_LEN,
            kv_seq_len=SEQ_LEN,
            num_q_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
        )

        append_spec = ExperimentSpec(
            workload_type="append",
            batch_size=BATCH_SIZE,
            q_seq_len=APPEND_Q_CHUNK,
            kv_seq_len=SEQ_LEN,
            num_q_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
        )

        decode_spec = ExperimentSpec(
            workload_type="decode",
            batch_size=BATCH_SIZE,
            q_seq_len=1,
            kv_seq_len=SEQ_LEN,
            num_q_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
        )

        specs.append((model_name, "prefill", prefill_spec))
        specs.append((model_name, "append", append_spec))
        specs.append((model_name, "decode", decode_spec))

    return specs


def run_benchmark(framework_name, exp_spec, num_warmup=3, num_iters=20):
    """Run a single benchmark."""
    try:
        runner_class = FRAMEWORKS[framework_name]
        runner = runner_class()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        result = runner.run(exp_spec, num_warmup=num_warmup, num_iters=num_iters)

        kernel_time_ms = result.kernel_time / 1_000_000
        return kernel_time_ms, None

    except Exception as e:
        return None, str(e)
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"\nConfiguration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Append Q chunk: {APPEND_Q_CHUNK}")
    print(f"\n{'=' * 100}\n")

    specs = create_experiment_specs()
    results = {}

    total = len(FRAMEWORKS) * len(specs)
    current = 0

    for framework_name in FRAMEWORKS.keys():
        results[framework_name] = {}

        for model_name, workload_type, exp_spec in specs:
            current += 1
            key = f"{model_name}-{workload_type}"

            print(f"[{current}/{total}] {framework_name:15} | {model_name:15} | {workload_type:8}", end=" ... ")

            latency_ms, error = run_benchmark(framework_name, exp_spec)

            if error:
                print(f"ERROR: {error[:50]}")
                results[framework_name][key] = None
            else:
                print(f"{latency_ms:.2f} ms")
                results[framework_name][key] = latency_ms

    print(f"\n{'=' * 100}")
    print("RESULTS TABLE")
    print(f"{'=' * 100}\n")

    print(f"{'Framework':<15}", end="")
    for model_name, _ in [("llama-3.1-8b", None), ("qwen3-30b-a3b", None), ("qwen3-235b-a22b", None)]:
        for wl in ["prefill", "append", "decode"]:
            col_name = f"{model_name.split('-')[0]}-{wl[:3]}"
            print(f" | {col_name:>10}", end="")
    print()
    print("-" * 100)

    for framework_name in FRAMEWORKS.keys():
        print(f"{framework_name:<15}", end="")
        for model_name in ["llama-3.1-8b", "qwen3-30b-a3b", "qwen3-235b-a22b"]:
            for wl in ["prefill", "append", "decode"]:
                key = f"{model_name}-{wl}"
                val = results[framework_name].get(key)
                if val is None:
                    print(f" | {'ERROR':>10}", end="")
                else:
                    print(f" | {val:>10.2f}", end="")
        print()

    print(f"\n{'=' * 100}")

    output_dir = Path("./benchmark_results/readme")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "readme_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
