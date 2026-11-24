# squid5060ti

**A 16 GB zero-spill fork of [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090) — single-binary, zero-Python Zig + Blackwell CUDA (`sm_120a`) inference engine for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) tuned for 16 GB Blackwell GPUs (RTX 5060 Ti 16 GB, RTX 5070 Ti, RTX 5080).**

On a single **RTX 5060 Ti 16 GB** (`36` SMs, `376.4 GB/s` measured D2D BW, Windows 11 + WSL2 Ubuntu 26.04):
- **`125.9–132.6 tok/s`** single-user mixed decode (`151–174 tok/s` on code — **2.8–3.8× `llama.cpp`**)
- **`110.4–111.0 tok/s`** non-speculative (`--mtp 0`, **83.0%** of measured GDDR7 bandwidth ceiling)
- **`1,950–2,061 tok/s`** cold 16K prefill (**5.4–5.7× `llama.cpp`**)
- **`48K–64K` context** in **`14.48–14.63 GiB` VRAM** (`~501 MiB` free under Windows WDDM at 48K, **0 GiB host-RAM spill**)
- **3/3 verbatim needle recall** verified at **25.7K** (`165.5 tok/s` decode) and **43.9K** (`119.3 tok/s` decode) context

---

## Benchmark: RTX 5060 Ti 16 GB (`c=1`, `bench.py` 6-prompt panel, `temp=0.6`)

| Metric | `llama.cpp` (`drofn-16GB-96K` GGUF) | `squid5060ti` (`NVFP4 + Q4 KV`) | Speedup | Measured Ceiling |
|---|---:|---:|---:|---:|
| **c1 decode (median of 6)** | `45.61 tok/s` | **`125.9–132.6 tok/s`** *(code: `151–174`)* | **2.8–3.8×** | `160–200 tok/s` |
| **c1 decode (`mtp=0`, greedy)** | — | **`110.4–111.0 tok/s`** | **2.4×** | `133 tok/s` (83.0%) |
| **TTFT (short prompt)** | `254.6 ms` | **`78.0–82.9 ms`** | **3.1×** | — |
| **Prefill ~4K** (3,300 tok) | `411.0 tok/s` | **`1,629–1,755 tok/s`** | **4.0–4.3×** | `3,649 tok/s` |
| **Prefill ~16K** (12,898 tok) | `362.5 tok/s` | **`1,950–2,061 tok/s`** | **5.4–5.7×** | `3,359 tok/s` (61.4%) |
| **3-Needle Gate (25.7K / 43.9K)** | — | **3/3 PASS** (`2,003` / `1,879` pf) | — | — |
| **Peak VRAM (`nvidia-smi`)** | `15,934 / 16,311 MiB` | **`15,550 / 16,311 MiB`** (`501 MiB` free @ 48K) | **-384 MiB** | ≤ `15.4 GiB` cap |

Full hardware roofline math, interleaved A/B hill-climb log (including rejected levers), and raw ledger are in [`THEORY-5060TI.md`](THEORY-5060TI.md), [`PLAN-5060TI.md`](PLAN-5060TI.md), and [`ledger-5060ti.jsonl`](ledger-5060ti.jsonl).

---

## Why `squid5090` needed changes for 16 GB Blackwell GPUs

Upstream `squid5090` targets a 32 GiB RTX 5090 and uses `15.59 GiB` at 8K context (plus an `879 MiB` load-time staging spike), which OOMs or triggers severe WDDM host-RAM spill on 16 GB cards (`15.93 GiB` total, `~15.1–15.4 GiB` free on Windows). `squid5060ti` makes five targeted changes:

1. **Host-pinned `embed.weight` by default** in `nvfp4`/`fp8` mode (`-0.86 GiB` VRAM; 4.6 KB/token over PCIe).
2. **Host-pinned `markov_w1`** (`-98 MiB` VRAM; 512 B/draft step over PCIe).
3. **Zero-copy host-pinned load staging (`d_stage`, `d_fold_bf16`)** (`-879 MiB` peak load VRAM spike eliminated).
4. **`prefill_batch_max = 512`** instead of `2048` (`-515 MiB` runtime VRAM; saturates 36–84 SMs on 16 GB Blackwell).
5. **Default `--kv q4` and `--spec-threshold 0.90`** (tuned via interleaved A/B on 376.4 GB/s GDDR7).

---

## Quickstart (Linux & Windows WSL2)

### Requirements
- **GPU:** 16 GB+ NVIDIA Blackwell (`sm_120a`: RTX 5060 Ti 16 GB, RTX 5070 Ti, RTX 5080, RTX 5090)
- **CUDA:** `13.1+` (`13.3` recommended on Ubuntu 26.04 / glibc 2.43)
- **Zig:** `0.17.1+`
- **Host C++:** `g++` (on Ubuntu 26.04 with gcc-15 default, install `g++-14` and pass `-Dccbin=g++-14`)

### Build
```bash
git clone https://github.com/omerdikyol/squid5060ti.git
cd squid5060ti

# Linux / Ubuntu 24.04
zig build -Doptimize=ReleaseFast

# Ubuntu 26.04 (gcc-15 host default)
zig build -Doptimize=ReleaseFast -Dccbin=g++-14
```

### Download Weights
```bash
uvx --from huggingface-hub hf download thinkingmachines/Inkling-2-Mini \
  --local-dir ~/models/inkling-2-mini
```

### Serve (48K context, ~500 MiB WDDM headroom on Windows + WSL2)
```bash
./zig-out/bin/squid5060ti serve ~/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 49152 --mtp 7 --spec-threshold 0.90
```

*(On dedicated Linux without a desktop compositor, `--ctx 65536` fits cleanly with ~1.2 GiB free.)*

### Windows `llama-swap` Integration (via WSL2)
```yaml
models:
  inkling-2-mini-squid:
    name: "Inkling 2 Mini 26B (squid5060ti Zig+sm_120a NVFP4)"
    cmd: wsl.exe -d Ubuntu -- env LD_LIBRARY_PATH=/usr/local/cuda/lib64 /home/<user>/squid5060ti/zig-out/bin/squid5060ti serve /home/<user>/models/inkling-2-mini --host 127.0.0.1 --port ${PORT} --mode nvfp4 --kv q4 --ctx 49152 --mtp 7 --spec-threshold 0.90
    checkEndpoint: /health
```

---

## Attribution
Forked from [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090). Optimized using [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill). See [`NOTICE`](NOTICE) for full upstream attributions.
