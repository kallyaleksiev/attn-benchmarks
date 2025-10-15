"""Model and experiment configurations."""

import abc
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class ModelConfig:
    """Model architecture configuration (attention-specific parameters only)."""

    name: str
    num_attention_heads: int  # Q heads
    num_key_value_heads: int  # KV heads (Q groups for GQA)
    head_dim: int

    def __repr__(self):
        return f"{self.name} (H_q={self.num_attention_heads}, H_kv={self.num_key_value_heads}, D={self.head_dim})"


WorkloadType = Literal["prefill", "decode", "append"]


@dataclass
class ExperimentSpec:
    """Specification for tensor shapes and dtype in a benchmark experiment.

    NOTE: for now dtype is just bf16
    """

    workload_type: WorkloadType

    batch_size: int

    q_seq_len: int
    kv_seq_len: int

    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    # NOTE: For now just bf16
    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def __repr__(self):
        return (
            f"{self.workload_type} {self.dtype} "
            f"(B={self.batch_size}, Q={self.q_seq_len}, KV={self.kv_seq_len}, "
            f"H_q={self.num_q_heads}, H_kv={self.num_kv_heads}, D={self.head_dim})"
        )


LLAMA_3_1_8B = ModelConfig(
    name="llama-3.1-8b",
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
)


QWEN3_30B_A3B = ModelConfig(
    name="qwen3-30b-a3b",
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
)


QWEN3_235B_A22B = ModelConfig(
    name="qwen3-235b-a22b",
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
)

_ALL_MODELS_CONFIGS = [
    LLAMA_3_1_8B,
    QWEN3_30B_A3B,
    QWEN3_235B_A22B,
]


_BATCH_SIZES = [
    1,
    256,
]

_SEQ_LENS = [
    512,
    4096,
    16384,
]

_APPEND_SEQ_LENS = [
    (8, 512),
    (128, 4096),
    (512, 16384),
]


def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name."""
    for model in _ALL_MODELS_CONFIGS:
        if model.name == name:
            return model
    raise ValueError(f"Unknown model: {name}. Available: {[m.name for m in _ALL_MODELS_CONFIGS]}")


def get_prefill_configs() -> list[ExperimentSpec]:
    all_configs = []

    for model_config in _ALL_MODELS_CONFIGS:
        for prefill_seq_len in _SEQ_LENS:
            for batch_size in _BATCH_SIZES:
                all_configs.append(
                    ExperimentSpec(
                        workload_type="prefill",
                        batch_size=batch_size,
                        q_seq_len=prefill_seq_len,
                        kv_seq_len=prefill_seq_len,
                        num_q_heads=model_config.num_attention_heads,
                        num_kv_heads=model_config.num_key_value_heads,
                        head_dim=model_config.head_dim,
                    )
                )

    return all_configs


def get_decode_configs() -> list[ExperimentSpec]:
    all_configs = []

    for model_config in _ALL_MODELS_CONFIGS:
        for decode_seq_len in _SEQ_LENS:
            for batch_size in _BATCH_SIZES:
                all_configs.append(
                    ExperimentSpec(
                        workload_type="decode",
                        batch_size=batch_size,
                        q_seq_len=1,
                        kv_seq_len=decode_seq_len,
                        num_q_heads=model_config.num_attention_heads,
                        num_kv_heads=model_config.num_key_value_heads,
                        head_dim=model_config.head_dim,
                    )
                )

    return all_configs


def get_append_configs() -> list[ExperimentSpec]:
    all_configs = []

    for model_config in _ALL_MODELS_CONFIGS:
        for append_q_len, append_kv_len in _APPEND_SEQ_LENS:
            for batch_size in _BATCH_SIZES:
                all_configs.append(
                    ExperimentSpec(
                        workload_type="append",
                        batch_size=batch_size,
                        q_seq_len=append_q_len,
                        kv_seq_len=append_kv_len,
                        num_q_heads=model_config.num_attention_heads,
                        num_kv_heads=model_config.num_key_value_heads,
                        head_dim=model_config.head_dim,
                    )
                )

    return all_configs


def get_all_exp_configs() -> list[ExperimentSpec]:
    return get_prefill_configs() + get_decode_configs() + get_append_configs()


Framework = Literal["flashinfer"]


@dataclass(frozen=True)
class ExperimentResult:
    exp_spec: ExperimentSpec
    """The experiment configuration."""

    framework: Framework
    """Which framework was used."""

    o: torch.tensor
    """Actual output."""

    preprocess_time: int
    """seconds spent preprocessing."""

    kernel_time: int
    """seconds spent running computation."""


class ExperimentRunner(abc.ABC):
    framework: Framework
    """Which framework was used."""

    @abc.abstractmethod
    def preprocess(
        self,
        exp_spec: ExperimentSpec,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Create data for the experiment and do the necessary preprocessing, e.g.
        reshaping and so on.
        """

    @abc.abstractmethod
    def compute_attention(
        self,
        q: torch.tensor,
        k: torch.tensor,
        v: torch.tensor,
    ) -> torch.tensor:
        """Run the actual kernel."""

    @abc.abstractmethod
    def run(
        self,
        exp_spec: ExperimentSpec,
        num_warmup: int = 3,
        num_iters: int = 20,
    ) -> ExperimentResult:
        """Run preprocessing and compute attention to produce an experiment result."""
