# Attention Benchmarks

**H200 BF16 latency:**

batch_size=256
seq_len=4096
append_q_chunk=128

| Framework     | Llama 3-8b-prefill | Llama 3-8b-append | Llama 3-8b-decode | Qwen3-30b-prefill | Qwen3-30b-append | Qwen3-30b-decode | Qwen3-235b-prefill | Qwen3-235b-append | Qwen3-235b-decode |
| ------------- | ------------------ | ----------------- | ----------------- | ----------------- | ---------------- | ---------------- | ------------------ | ----------------- | ----------------- |
| flash attn    | 188.00             | 6.59              | 1.61              | 188.42            | 6.67             | 0.90             | 377.61             | 13.36             | 0.85              |
| flashinfer    | 139.01             | 7.23              | 1.62              | 141.16            | 7.24             | 1.52             | 278.40             | 12.63             | ERROR             |
| te            | 117.70             | 3.74              | 1.19              | 117.66            | 3.63             | 0.71             | 239.45             | 7.55              | 0.72              |
| torch flex    | OOM                | 141.33            | 30.76             | OOM               | 140.86           | 30.30            | OOM                | OOM               | 58.97             |
| torch compile | 217.03             | 14.93             | 6.54              | 213.95            | 13.06            | 6.23             | 432.18             | 25.57             | 9.95              |
| torch naive   | OOM                | 43.07             | 21.34             | OOM               | 42.58            | 20.93            | OOM                | OOM               | 41.19             |
| torch sdpa    | OOM                | 23.52             | 23.38             | OOM               | 23.11            | 22.95            | OOM                | 45.90             | 45.73             |
