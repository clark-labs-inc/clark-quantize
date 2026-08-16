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
