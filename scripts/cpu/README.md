# CPU Quant Research Node

The CPU bootstrap provisions an x86-64 node for packing, scalar-oracle,
microkernel, llama.cpp integration, and bounded Qwen evaluation work. It does
not install CUDA-only packages or claim that a small CPU node can train the
27B parent model.

The repository must already be present at `/workspace/clark-quantize`. Run:

```bash
sudo bash scripts/cpu/bootstrap_quant_cpu_node.sh
source /workspace/venvs/quant-cpu/bin/activate
```

The script installs CPU-only PyTorch in an isolated Python 3.13 environment,
pins and builds llama.cpp, creates persistent model/run/cache directories, and
writes `/workspace/quant-state/cpu_environment.json`.

Fetch the revision-pinned 4B development parent and 27B research parent with:

```bash
bash scripts/cpu/fetch_qwen_parents.sh
```

The fetch is resumable, refuses concurrent writers, verifies that every shard
referenced by each safetensors index exists, and writes the compact receipt
`/workspace/quant-state/qwen_parent_models.json`.

Primary paths:

| Path | Purpose |
| --- | --- |
| `/workspace/clark-quantize` | source checkout |
| `/workspace/models` | Hugging Face parents and packed models |
| `/workspace/quant-runs` | run outputs and compact checkpoints |
| `/workspace/quant-state` | environment and launch receipts |
| `/workspace/hf-cache` | Hugging Face cache |
| `/workspace/src/llama.cpp` | pinned runtime source |
| `/workspace/build/llama.cpp` | native optimized build |

The default thread count is four because the first Hetzner CCX33 reference
node exposes four physical Milan cores and eight SMT threads. Benchmarks must
report physical-core and SMT runs separately.

After the parents are fetched, the bounded 27B CPU packing lane is:

```bash
python experiments/qwen38_27b_spot/download_model.py --workers 4
python experiments/qwen38_27b_spot/preflight_quantization.py --device cpu
python experiments/qwen38_27b_spot/base3_packing_research.py --device cpu
python experiments/qwen38_27b_spot/full_base3_pack.py \
  --device cpu --workers 4 --max-in-flight 4 --row-batch 64
```

`download_model.py` is idempotent here: it verifies the already-fetched pinned
shards and creates the legacy download receipt required by the original spot
pipeline. The full pack is chunked and resumable; CPU execution does not weaken
its code, scale, source-identity, or independent-decode gates.

Build and test the standalone ternary CPU primitives with:

```bash
bash scripts/cpu/build_ternary_cpu.sh
python -m unittest clark_quantize.test_ternary_cpu
```

After the full 27B artifact has promoted, run the sealed real-matrix K2 gate:

```bash
python experiments/llm_ternary_158bit_dossier/cpu_runtime/benchmark_ternary_gemv.py
```

The benchmark authenticates codes and FP16 scales from the artifact, evicts a
256 MiB cache buffer between samples, pins persistent workers to distinct
physical cores, compares against a checksumed STREAM-like read control, and
writes `/workspace/quant-state/qwen38-ternary-gemv-avx2-v5.json`.
