"""FlashInfer attention implementation http://flashinfer.ai/

NOTE: It uses page layour APIs for all workloafs:
- BatchDecodeWithPagedKVCacheWrapper for decode (q_len=1)
- BatchPrefillWithPagedKVCacheWrapper for prefill/append (q_len>1)
"""

from dataclasses import dataclass
from typing import Optional, override

import flashinfer
import torch

from ..exp_config import ExperimentResult, ExperimentRunner, ExperimentSpec


@dataclass
class PagedKVCache:
    """Paged KV cache with all metadata for FlashInfer."""

    k_paged: torch.Tensor  # (num_pages, page_size, num_kv_heads, head_dim)
    v_paged: torch.Tensor  # (num_pages, page_size, num_kv_heads, head_dim)
    indptr: torch.Tensor  # Page table indptr (batch_size + 1,)
    indices: torch.Tensor  # Page indices (num_pages,)
    last_page_len: torch.Tensor  # Last page length per sequence (batch_size,)
    page_size: int


@dataclass
class PreprocessedData:
    """All data prepared during preprocessing, ready for kernel execution."""

    q: torch.Tensor  # Query: (batch_size, num_q_heads, head_dim) for decode, (batch_size, q_len, num_q_heads, head_dim) for prefill
    kv_cache: PagedKVCache  # Paged KV cache
    qo_indptr: Optional[torch.Tensor]  # Query indptr for prefill (batch_size + 1,), None for decode
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    batch_size: int
    q_len: int


class FlashInferRunner(ExperimentRunner):
    def __init__(self):
        self.framework = "flashinfer"
        self._workspace_buffer: Optional[torch.Tensor] = None
        self._prefill_wrapper: Optional[flashinfer.BatchPrefillWithPagedKVCacheWrapper] = None
        self._decode_wrapper: Optional[flashinfer.BatchDecodeWithPagedKVCacheWrapper] = None
        self._current_exp_spec: Optional[ExperimentSpec] = None
        self._preprocessed_data: Optional[PreprocessedData] = None

    def _get_workspace_buffer(self, device: str = "cuda") -> torch.Tensor:
        """Get or create workspace buffer (256MB)."""
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                256 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        return self._workspace_buffer

    def _get_prefill_wrapper(self, device: str = "cuda"):
        """Get or create prefill wrapper."""
        if self._prefill_wrapper is None:
            workspace = self._get_workspace_buffer(device)
            self._prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD", use_cuda_graph=False
            )
        return self._prefill_wrapper

    def _get_decode_wrapper(self, device: str = "cuda"):
        """Get or create decode wrapper."""
        if self._decode_wrapper is None:
            workspace = self._get_workspace_buffer(device)
            self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD", use_cuda_graph=False
            )
        return self._decode_wrapper

    def _create_paged_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        kv_len: int,
        page_size: int = 16,
    ) -> PagedKVCache:
        """Convert K, V tensors to paged KV cache format."""
        device = k.device
        dtype = k.dtype

        # K, V should be (batch_size, kv_len, num_kv_heads, head_dim)
        num_kv_heads = k.shape[2]
        head_dim = k.shape[3]

        # Calculate paging
        num_pages_per_seq = (kv_len + page_size - 1) // page_size
        total_pages = batch_size * num_pages_per_seq

        # Allocate paged cache
        k_paged = torch.zeros(
            total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_paged = torch.zeros(
            total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Fill pages
        indices_list = []
        last_page_len_list = []

        for batch_idx in range(batch_size):
            for page_idx in range(num_pages_per_seq):
                global_page_idx = batch_idx * num_pages_per_seq + page_idx
                start_idx = page_idx * page_size
                end_idx = min(start_idx + page_size, kv_len)
                page_len = end_idx - start_idx

                # Copy data to page
                k_paged[global_page_idx, :page_len] = k[batch_idx, start_idx:end_idx]
                v_paged[global_page_idx, :page_len] = v[batch_idx, start_idx:end_idx]

                indices_list.append(global_page_idx)

            # Last page length for this sequence
            last_page_len_list.append((kv_len - 1) % page_size + 1)

        # Create metadata tensors
        indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device=device,
        )
        indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
        last_page_len = torch.tensor(last_page_len_list, dtype=torch.int32, device=device)

        return PagedKVCache(
            k_paged=k_paged,
            v_paged=v_paged,
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            page_size=page_size,
        )

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
            # === DECODE: q_len = 1 ===
            # Q: (batch_size, num_q_heads, head_dim)
            q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)

            # K, V: (batch_size, kv_len, num_kv_heads, head_dim)
            k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            # Convert to paged format
            kv_cache = self._create_paged_kv_cache(k, v, batch_size, kv_len)

            # No qo_indptr for decode
            qo_indptr = None

        else:
            # === PREFILL/APPEND: q_len > 1 ===
            # Q: (batch_size, q_len, num_q_heads, head_dim)
            q = torch.randn(batch_size, q_len, num_q_heads, head_dim, dtype=dtype, device=device)

            # K, V: (batch_size, kv_len, num_kv_heads, head_dim)
            k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            # Convert to paged format
            kv_cache = self._create_paged_kv_cache(k, v, batch_size, kv_len)

            # Create qo_indptr for prefill
            qo_indptr = torch.arange(
                0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
            )

        # Store preprocessed data
        self._preprocessed_data = PreprocessedData(
            q=q,
            kv_cache=kv_cache,
            qo_indptr=qo_indptr,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            batch_size=batch_size,
            q_len=q_len,
        )

        # Return for signature compatibility (not used downstream)
        return q, k, v

    @override
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        assert self._current_exp_spec is not None, "Must call run() first"
        assert self._preprocessed_data is not None, "Must call preprocess() first"

        exp_spec = self._current_exp_spec
        data = self._preprocessed_data
        device = data.q.device

        if exp_spec.workload_type == "decode":
            # DECODE
            wrapper = self._get_decode_wrapper(device)

            wrapper.plan(
                indptr=data.kv_cache.indptr,
                indices=data.kv_cache.indices,
                last_page_len=data.kv_cache.last_page_len,
                num_qo_heads=data.num_q_heads,
                num_kv_heads=data.num_kv_heads,
                head_dim=data.head_dim,
                page_size=data.kv_cache.page_size,
                q_data_type=data.q.dtype,
            )

            output = wrapper.run(data.q, (data.kv_cache.k_paged, data.kv_cache.v_paged))

            return output

        else:
            # PREFILL/APPEND
            wrapper = self._get_prefill_wrapper(device)

            # Flatten Q for prefill wrapper: (batch_size, q_len, H, D) -> (batch*q_len, H, D)
            q_flat = data.q.reshape(-1, data.num_q_heads, data.head_dim)

            wrapper.plan(
                qo_indptr=data.qo_indptr,
                paged_kv_indptr=data.kv_cache.indptr,
                paged_kv_indices=data.kv_cache.indices,
                paged_kv_last_page_len=data.kv_cache.last_page_len,
                num_qo_heads=data.num_q_heads,
                num_kv_heads=data.num_kv_heads,
                head_dim_qk=data.head_dim,
                page_size=data.kv_cache.page_size,
                q_data_type=data.q.dtype,
            )

            output_flat = wrapper.run(q_flat, (data.kv_cache.k_paged, data.kv_cache.v_paged))

            output = output_flat.reshape(
                data.batch_size, data.q_len, data.num_q_heads, data.head_dim
            )

            return output

    @override
    def run(
        self,
        exp_spec: ExperimentSpec,
        num_warmup: int = 3,
        num_iters: int = 20,
    ) -> ExperimentResult:
        self._current_exp_spec = exp_spec

        # === Phase 1: Preprocessing (timed separately) ===
        preprocess_start = torch.cuda.Event(enable_timing=True)
        preprocess_end = torch.cuda.Event(enable_timing=True)

        preprocess_start.record()
        q, k, v = self.preprocess(exp_spec)
        preprocess_end.record()
        torch.cuda.synchronize()

        preprocess_time_ms = preprocess_start.elapsed_time(preprocess_end)

        # === Phase 2: Warmup ===
        for _ in range(num_warmup):
            _ = self.compute_attention(q, k, v)
            torch.cuda.synchronize()

        # === Phase 3: Kernel timing (main measurement) ===
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

        # Use median kernel time
        import statistics

        kernel_time_ms = statistics.median(kernel_times)

        return ExperimentResult(
            exp_spec=exp_spec,
            framework=self.framework,
            o=output,
            preprocess_time=int(preprocess_time_ms * 1_000_000),  # Convert to nanoseconds
            kernel_time=int(kernel_time_ms * 1_000_000),  # Convert to nanoseconds
        )
