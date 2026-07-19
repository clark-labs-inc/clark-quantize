# Prism ML: The People, the Papers, and the Reconstructed Thinking

**Date:** 2026-07-03. Sources: GitHub/HF APIs (commit authors, org data, release timeline), prismml.com, PR Newswire/press coverage, arXiv, both Bonsai whitepapers, and our own weight forensics (`bonsai_ternary_deep_forensics.md`). Verification sweep: deep-research workflow (cited claims merged below).

---

## 1. Who they are

**Founders** (per launch press, 2026-03-31 stealth exit; $16.25M seed from Khosla Ventures + Cerberus, compute grants from Google and Caltech):

| Person | Role | Background |
|---|---|---|
| **Babak Hassibi** | Founder & CEO | Caltech EE professor (Bohn Professor). Stanford PhD under Thomas Kailath. **Co-author of Optimal Brain Surgeon (NeurIPS 1992/93)** — the canonical second-order pruning paper. Three decades of estimation/information theory (H∞, Kalman, CGMT-style precise high-dimensional analysis), then the SMD implicit-bias program. |
| **Sahin Lale** | Co-founder, **Co-Head of Research** | Caltech EE PhD 2023, thesis "Learning and Control of Dynamical Systems" — advisor **Anima Anandkumar**, committee incl. Hassibi (Scholar affiliation now "PrismML, Caltech"). Regret-optimal & competitive-ratio control, Thompson-sampling LQR, KCRL, Fourier Neural Operators, and an *analog arrhythmia-classifier chip* (EKGNet) — the constrained-hardware sensibility. Co-author of "Explicit Regularization via Regularizer Mirror Descent" (2022) with Azizan & Hassibi. |
| **Omead Pooladzandi** | Co-founder | UCLA PhD (Pottie): **PSGD / Lie-group preconditioners** (with Xi-Lin Li), second-order coresets (with Mirzasoleiman), poison purification via EBMs, **"Scaling Behavior of Discrete Diffusion Language Models"** (Dec 2025, with Schölkopf/Hofmann group), and co-author of the pre-stealth capstone below with Hassibi's students. Co-authored a 2023 synthetic-data paper **with Pasha Khosravi** — the recruiting edge. |
| **Reza Sadri** | Founder & VP Strategy | Industry veteran (storage/data infrastructure exec lineage) — the commercial operator. |

**Engineering (from commit forensics):**
- **Pasha Khosravi** (`khosravipasha`, HF `pashak`) — UCLA StarAI PhD (Guy Van den Broeck; probabilistic circuits: expected predictions, missing-data reasoning, Juice.jl), ex-LinkedIn applied research. At Prism he personally authored the GGUF **Q2_0 type + CPU/Metal/CUDA kernels** (Apr 18–19), uploads every HF artifact, and drove the Q1_0 **upstreaming into mainline llama.cpp** (PRs #21273/#21528/#21629/#21636; Vulkan #21539 was community). |
- **"Brian"** (`bri-prism`, account created 2026-05-27) — kernels specialist: AVX-512-VNNI Q1_0/Q2_0 dot products (Jun 4), **Hopper sm_90a wgmma prefill paths for Q1_0/Q2_0** (Jun 10–12), Blackwell gating. Identity not public.
- Community satellites: Agneya Tharun (Vulkan Q2_0), pl752 (x86 q1_0), Evan Walters (PSGD-Kron co-author of Pooladzandi's circle — cameo commits on the image demos), Filip Strand (upstream mflux author; Prism forked mflux for image deployment). HF org: 11 members. Likely-but-unconfirmed additional researchers: Hassibi students **Danil Akhtiamov** and **Reza Ghane** (authors on every Prism-relevant theory paper; no public Prism affiliation found).

## 2. The papers that matter (the intellectual supply chain)

**The deep root (1993):** Hassibi & Stork, *Optimal Brain Surgeon* — remove the weights whose deletion costs least, judged by the loss Hessian. GPTQ/OBQ descend from it. **The ternary zero-state is OBS reborn**: Bonsai's 37–40% zeros are learned pruning, and the brand ("Bonsai" = a carefully pruned tree) says they know it.

**The theory program (Hassibi lab, 2019→2026):**
- Azizan & Hassibi, *Stochastic Mirror Descent* implicit-regularization line (2019–2021); Azizan, **Lale**, Hassibi, *Explicit Regularization via Regularizer Mirror Descent* (2022) — SGD-family algorithms are mirror descent; the mirror map dictates which interpolator you converge to.
- Akhtiamov, Ghane, Hassibi: *A Novel Gaussian Min-Max Theorem* (2024) — the precise-analysis toolkit; *One-Bit Quantization and Sparsification for Multiclass Linear Classification* (Feb 2024); **One-Bit Quantization for Random Features Models** (Oct 2025): *"quantizing weights of all layers except the last incurs no loss in generalization error"* — plus laptop-GPU speedup demos inside a theory paper (the deployment instinct was already there).
- Akhtiamov, Ghane, **Pooladzandi**, Hassibi, **Implicit Bias and Convergence of Matrix Stochastic Mirror Descent** (Feb 2026, one month pre-stealth): matrix SMD converges exponentially to the interpolator **minimizing Bregman divergence from initialization**.
- Ghane, Akhtiamov, Hassibi, *Dual Space Preconditioning for Gradient Descent in the Overparameterized Regime* (Mar 2026) — precondition in the dual/mirror space (the PSGD bridge).
- Han, Kostina, Hassibi, *Quantized Online LQR* (Apr 2026) — quantization meets control: the robotics tell.

**The practice adopted:** BitNet (2310.11453) and BitNet b1.58 (2402.17764) — both cited in the whitepapers; our forensics prove the shipped quantizer IS b1.58's per-group-128 absmean rule. Direct public prior art for the *conversion* recipe: **MSR's BitNet Distillation (arXiv 2510.13998, Oct 2025)**, which distills off-the-shelf checkpoints (naming Qwen) into 1.58-bit — Prism's contribution is the industrialization (kernels, formats, benchmark discipline, three scales, image models), not the core trick.

**Format roadmap (verified, from their own PR threads):** after ggerganov requested a group-size change, Prism opened mainline llama.cpp PRs #24448/#25188 for **group-64** Q2_0 — meaning mainline ternary GGUF will be 2.25 b/w, while the tighter g128 lives on as fork-only **PQ2_0** (Khosravi, PR #32: "probably going to make the official Q2_0 to be group size 64, and then have PQ2_0 with group size 128 in our fork").

**The adjacent bets:** Pooladzandi's discrete-diffusion scaling (Dec 2025) → bonsai-image (May 2026); PSGD/Lie-group preconditioners → their training efficiency edge (undisclosed, but the dual-space-preconditioning paper is the tell).

## 3. The reconstructed thinking

**The core belief, stated as theory→product:** extreme weight discretization is *nearly free* in the overparameterized regime (one-bit RF paper), and the training dynamics that get you there are not a hack but a mirror-descent process whose mirror map you can choose (SMD program). STE-on-master-weights — what BitNet does and what we proved they ship — *is* mirror descent: the fp32 master is the dual variable, the quantizer is the link function. Hassibi's lab spent seven years building the theory of exactly this object before founding a company to industrialize it.

**Theory predicts our forensics.** Matrix SMD converges to the interpolator Bregman-closest to initialization — and the shipped weights sit measurably close to base: 91% sign inheritance, 77% top-k support overlap, embeddings bit-frozen at PTQ, head 85% unmoved. The one-bit paper's "all layers except the last" carve-out mirrors the practice of treating vocab-adjacent matrices differently. What looked like engineering shortcuts in our forensics is the theory's fingerprint.

**Why Qwen3 bases:** best open Apache-compatible quality-per-size family at 1.7/4/8B with GQA + QK-norm (quantization-friendly), tied-embed small variants, and a vocab whose tail they could prune. They convert rather than pretrain: $16M seed buys KD conversions (10s of B tokens), not 4T-token pretrains — intelligence *density* is the axis where a seed-stage company can beat Qwen/Meta, hence the −ln(1−score/100)/GB metric invented to make that legible.

**Why deployment-first:** the founding team is unusual — a theorist-CEO whose lab measured laptop speedups, a control theorist who taped out an analog classifier chip, an optimizer specialist, a storage-industry operator, and an engineer-lead who upstreams kernels into llama.cpp mainline. Their moat thesis is *format + kernels + theory*, not weights (weights are Apache 2.0): Q1_0 is already in mainline llama.cpp; Q2_0/PQ2_0 kernels live in their fork across NEON/Metal/CUDA/Vulkan/VNNI/wgmma; MLX PR pending. Giving weights away seeds the format war they intend to win.

**Release cadence (reconstructed from HF/GitHub/PR timestamps):**
- Mar 18–29: 1-bit Bonsai 8B→4B/1.7B (GGUF + MLX + unpacked) — stealth exit Mar 31, press Apr 2–6
- Apr 13–18: Ternary family + llama.cpp Q2_0 kernels + whitepaper (Apr 16) + AWQ-4bit comparison artifact
- May 18–21: bonsai-image binary/ternary 4B (FLUX.2-klein QAT; gemlite CUDA + MLX 2-bit packs)
- Jun 4–12: AVX-512-VNNI + **Hopper wgmma** Q1/Q2 prefill + Blackwell gating; **PQ2_0** repack uploaded Jun 10 — the same week (PQ2_0 = kernel-layout permutation for VNNI/wgmma, same 1075.0 MB as Q2_0, not extra compression)

**Where they're heading (evidence-weighted):**
1. **Datacenter ternary serving** — Hopper/Blackwell prefill kernels make no sense for iPhones; "datacenters can't sustain them" is on the homepage. Expect cloud-side dense-batch ternary (possibly sglang — they forked it in April).
2. **Robotics/agents on-device** — homepage says robotics; *Quantized Online LQR* is the academic advance party; Lale's control corpus is the bench.
3. **Bigger/newer bases** — the obvious next conversion targets are Qwen3.5's hybrid models (our current project is effectively racing them there) and possibly MoE (whitepaper cites LFM2-8B-A1B).
4. **Image → video diffusion** at low bit (Pooladzandi's discrete-diffusion scaling work; mflux fork).
5. **Sub-2-bit packing** — their README calls Q2_0 "at the cost of being larger than a tight ternary packing" and TQ1_0-style 1.6-bit "TBD"; Clark's pack40-log6-fp8 is *ahead* of them here.

**Open unknowns they never disclose:** KD corpus/token count, optimizer (dual-space preconditioning / PSGD-Kron is the educated bet given Pooladzandi + the Mar 2026 paper + Evan Walters' orbit), and training infra. No training-method paper exists — the whitepapers are deployment/benchmark documents; the method lives in the weights (where we read it) and in the theory papers (where they proved it).

## 4. Implications for our Qwen3.5 run

- Our forensically-derived recipe (absmean STE, frozen embed/norms, stay-close-to-init) is not just imitation — it matches the founders' own theory of why this works. Confidence upgrade.
- Their optimizer is likely a dual-space/Kron-preconditioned method; our Muon choice is the same family (orthogonalized ≈ Lie-group preconditioning). If the movement servo stalls, PSGD-Kron (Pooladzandi & Li 2024) is the literature-faithful alternative.
- They will predictably ship a Qwen3.5-based Bonsai; differentiation for Clark: hybrid-DeltaNet ternary first (they haven't touched linear-attention models), sub-2-bit packing (pack40 beats Q2_0's 2.125 b/w), and the signature-steering methodology itself.
