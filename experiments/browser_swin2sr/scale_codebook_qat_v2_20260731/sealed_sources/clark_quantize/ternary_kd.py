"""Signature-steered ternary KD: everything forensics fixed, plus controllers
for everything it couldn't recover.

Fixed by evidence (bonsai_ternary_deep_forensics.md):
  - AbsmeanTernaryLinear: BitNet-b1.58 forward (per-group-128 absmean scales
    recomputed from the live master, round-clip codes, STE). No learned scales.
  - ternarize_qwen35 / freeze map for the Qwen3.5 hybrid (DeltaNet islands,
    frozen embed/norms/vision/MTP).
  - SignatureMeter: measures the run against Prism's recovered fixed point
    (dequant movement, code churn vs PTQ init, zeros, sign inheritance).

Controllers for the unknowns:
  - KLBanditMixer: data mix ∝ measured per-arm teacher/student KL.
  - MovementServo: PI controller steering LR along the sqrt-diffusion
    movement reference; guardrails from forensic invariants.
  - should_stop(): budget-as-output stopping rule.

Losses: topk_kd_loss (vocab-safe KL), anchored_hidden_loss (full-attention
residual anchors for the 3:1 hybrid).
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GROUP = 128

# Prism fixed point, measured on Ternary-Bonsai-8B vs Qwen3-8B (trunk-weighted).
SIGNATURE_TARGET = {
    "movement": 0.71,   # ||dequant(W) - W_base|| / ||W_base||  (PTQ floor ~0.51)
    "churn": 0.30,      # fraction of codes changed vs absmean-PTQ init
    "zeros": 0.375,     # trained zero fraction
    "sign_inherit": 0.91,
}
SIGNATURE_BAND = {
    "movement": (0.65, 0.75),
    "churn": (0.28, 0.38),
    "zeros": (0.36, 0.41),
    "sign_inherit": (0.88, 0.93),
}
PTQ_MOVEMENT_FLOOR = 0.51
LEARNED_MOVEMENT_AT_BUDGET = math.sqrt(max(SIGNATURE_TARGET["movement"] ** 2 - PTQ_MOVEMENT_FLOOR ** 2, 0.0))


# ---------------------------------------------------------------------------
# Quantizer (fixed by evidence)
# ---------------------------------------------------------------------------

def absmean_quantize(w: Tensor, group: int = GROUP) -> tuple[Tensor, Tensor]:
    """BitNet-b1.58 per-group absmean ternarization. Returns (codes int8, scales)."""
    out, cin = w.shape
    if cin % group:
        raise ValueError(f"in_features {cin} not divisible by group {group}")
    wg = w.reshape(out, cin // group, group)
    s = wg.abs().mean(dim=-1).clamp_min(1e-8)
    codes = torch.round(wg / s.unsqueeze(-1)).clamp_(-1, 1)
    return codes.to(torch.int8), s


# Fused quantize-dequant. Triton path: one read of the fp32 master, one write
# of the bf16 dequant (~21 GB traffic for a 3.6B trunk vs ~80+ GB for the
# unfused op chain). Falls back to torch ops off-GPU or if compilation fails.
_TRITON_STATE = "untried"  # untried | ok | failed

try:  # pragma: no cover - exercised only on CUDA boxes
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as _libdevice

        _HAS_RINT = True
    except ImportError:
        _HAS_RINT = False

    @triton.jit
    def _absmean_dequant_kernel(w_ptr, q_ptr, numel, GPB: tl.constexpr, GROUP: tl.constexpr,
                                USE_RINT: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * GPB * GROUP + tl.arange(0, GPB * GROUP)
        mask = offs < numel
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        wg = tl.reshape(w, (GPB, GROUP))
        s = tl.sum(tl.abs(wg), axis=1) / GROUP
        s = tl.maximum(s, 1e-8)
        r = wg / s[:, None]
        if USE_RINT:
            q = _libdevice.rint(r)
        else:
            q = tl.floor(r + 0.5)  # half-up; differs from torch.round only on exact ties
        q = tl.minimum(tl.maximum(q, -1.0), 1.0) * s[:, None]
        tl.store(q_ptr + offs, tl.reshape(q, (GPB * GROUP,)).to(q_ptr.dtype.element_ty), mask=mask)

except ImportError:  # triton not installed (e.g. macOS dev box)
    _TRITON_STATE = "failed"


def _absmean_dequant_torch(w: Tensor, out_dtype: torch.dtype) -> Tensor:
    wg = w.reshape(w.shape[0], -1, GROUP)
    s = wg.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return (torch.round(wg / s).clamp_(-1, 1) * s).reshape_as(w).to(out_dtype)


@torch.no_grad()
def fused_absmean_dequant(w: Tensor, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    global _TRITON_STATE
    if not w.is_cuda or _TRITON_STATE == "failed":
        return _absmean_dequant_torch(w, out_dtype)
    q = torch.empty(w.shape, dtype=out_dtype, device=w.device)
    wc = w.contiguous()
    gpb = 8
    grid = (math.ceil(wc.numel() / (gpb * GROUP)),)
    try:
        _absmean_dequant_kernel[grid](wc.reshape(-1), q.reshape(-1), wc.numel(),
                                      GPB=gpb, GROUP=GROUP, USE_RINT=_HAS_RINT)
        _TRITON_STATE = "ok"
    except Exception as e:  # any launch/compile failure -> permanent torch fallback
        _TRITON_STATE = "failed"
        print(f"[ternary_kd] triton quantizer unavailable ({e}); using torch fallback")
        return _absmean_dequant_torch(w, out_dtype)
    return q


class _MasterSTE(torch.autograd.Function):
    """Return the cached dequant; route gradients straight through to the master."""

    @staticmethod
    def forward(ctx, master: Tensor, q: Tensor) -> Tensor:
        ctx.master_dtype = master.dtype
        return q

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad.to(ctx.master_dtype), None


class AbsmeanTernaryLinear(nn.Module):
    """Drop-in Linear whose forward is the exact Prism/BitNet quantizer + STE.

    The master weight is the only parameter. Scales are a deterministic
    function of the master (recomputed on master change) — the forensic proof
    is the frozen embedding, whose group scales equal absmean(base) at ratio
    1.000. There is nothing else to learn.

    Speed: the bf16 dequant is computed by a fused Triton kernel and cached,
    keyed on the master tensor's ``_version`` counter — in-place optimizer
    updates bump it, so the cache self-invalidates exactly once per optimizer
    step and is reused across all grad-accumulation micro-steps and probes.
    """

    def __init__(self, master: Tensor, bias: Optional[Tensor] = None) -> None:
        super().__init__()
        self.out_features, self.in_features = master.shape
        if self.in_features % GROUP:
            raise ValueError(f"in_features {self.in_features} % {GROUP} != 0")
        self.weight_master = nn.Parameter(master.detach().float().clone())
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().float().clone())
        else:
            self.register_parameter("bias", None)
        self._qcache: Optional[Tensor] = None
        self._qcache_version: int = -1

    def _cached_dequant(self, dtype: torch.dtype) -> Tensor:
        m = self.weight_master
        ver = m._version
        if (self._qcache is None or self._qcache_version != ver
                or self._qcache.dtype != dtype or self._qcache.device != m.device):
            self._qcache = fused_absmean_dequant(m.detach(), out_dtype=dtype)
            self._qcache_version = ver
        return self._qcache

    def quantized_weight(self, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        q = self._cached_dequant(dtype)
        if torch.is_grad_enabled() and self.weight_master.requires_grad:
            return _MasterSTE.apply(self.weight_master, q)
        return q

    @torch.no_grad()
    def codes_and_scales(self) -> tuple[Tensor, Tensor]:
        return absmean_quantize(self.weight_master.detach())

    def forward(self, x: Tensor) -> Tensor:
        w = self.quantized_weight(x.dtype if x.is_floating_point() else torch.bfloat16)
        b = None if self.bias is None else self.bias.to(w.dtype)
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group={GROUP}, rule=absmean"


# ---------------------------------------------------------------------------
# Model surgery: Qwen3.5 hybrid freeze/ternarize map (fixed by evidence)
# ---------------------------------------------------------------------------

TERNARIZE_SUFFIXES = (
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
)
HEAD_NAME = "lm_head"
EXCLUDE_SUBSTRINGS = ("visual", "vision", "mtp.", "in_proj_a", "in_proj_b", "conv1d")


def _parent_and_leaf(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


@torch.no_grad()
def ptq_freeze_embedding(embed: nn.Embedding) -> None:
    """Replace the embedding weight in place with its absmean-PTQ dequant.

    Forensic basis: Prism's embeddings are pure frozen PTQ (100% code match).
    Doing this before training means the student forward sees the exact
    deployed embedding (and, when tied, the exact deployed head) — no
    train/deploy mismatch, nothing to learn.
    """
    w = embed.weight.data
    codes, scales = absmean_quantize(w.float())
    deq = (codes.float() * scales.unsqueeze(-1)).reshape(w.shape)
    embed.weight.data = deq.to(w.dtype)
    embed.weight.requires_grad_(False)


def ternarize_qwen35(model: nn.Module, *, ptq_embed: bool = True,
                     train_norms: bool = False) -> dict[str, list[str]]:
    """Apply the forensic freeze/ternarize map. Returns {ternarized, head, frozen}.

    Everything not wrapped is frozen (requires_grad=False): embeddings, norms,
    DeltaNet islands (conv1d/A_log/dt_bias/in_proj_a/in_proj_b), MTP, vision.
    Embeddings are additionally PTQ'd in place (see ptq_freeze_embedding).

    Tied-embedding models (e.g. Qwen3.5-4B): lm_head shares the embedding
    tensor, so it is NOT wrapped — the PTQ'd embedding serves as the head,
    matching the frozen-PTQ signature of Prism's own tied 4B (token_embd
    zeros 0.3095 == the proven-frozen 8B embed signature).
    """
    embeds = [m for n, m in model.named_modules()
              if isinstance(m, nn.Embedding) and n.endswith("embed_tokens")]
    tied_ptrs = {e.weight.data_ptr() for e in embeds}

    wrapped: list[str] = []
    head: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in EXCLUDE_SUBSTRINGS):
            continue
        is_head = name.endswith(HEAD_NAME) or name == HEAD_NAME
        is_trunk = any(name.endswith(sfx) for sfx in TERNARIZE_SUFFIXES)
        if not (is_head or is_trunk):
            continue
        if is_head and module.weight.data_ptr() in tied_ptrs:
            continue  # tied head: handled by the PTQ'd embedding
        new = AbsmeanTernaryLinear(module.weight, module.bias)
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, new)
        (head if is_head else wrapped).append(name)
    frozen: list[str] = []
    for pname, p in model.named_parameters():
        if pname.endswith("weight_master") or (pname.endswith("bias") and "weight_master" in pname):
            continue
        owner = pname.rsplit(".", 1)[0]
        if any(owner == w or owner.startswith(w + ".") for w in wrapped + head):
            continue
        p.requires_grad_(False)
        frozen.append(pname)
    if ptq_embed:
        for e in embeds:
            ptq_freeze_embedding(e)
    trainable_norms: list[str] = []
    if train_norms:
        for pname, p in model.named_parameters():
            lower = pname.lower()
            if p.ndim != 1 or "norm" not in lower:
                continue
            if any(s in lower for s in ("visual", "vision", "mtp.")):
                continue
            p.requires_grad_(True)
            trainable_norms.append(pname)
    return {
        "ternarized": wrapped,
        "head": head,
        "frozen": frozen,
        "trainable_norms": trainable_norms,
    }


def collect_param_groups(model: nn.Module, *, master_lr: float, head_lr_scale: float = 0.25,
                         norm_lr: Optional[float] = None,
                         momentum: float = 0.95, ns_steps: int = 5,
                         state_dtype: torch.dtype = torch.bfloat16,
                         ns_dtype: torch.dtype = torch.bfloat16) -> list[dict]:
    """Param groups for BonsaiMuon: trunk masters (muon), lm_head master (muon, scaled).

    No scale group (scales aren't parameters), no prox_rho, no flip trust region —
    all three were falsified by the forensics. Defaults here run the momentum
    buffer and Newton-Schulz in bf16 (halves optimizer memory, runs NS on
    tensor cores); pass torch.float32 for the conservative variant.
    """
    trunk, head, norms = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("weight_master"):
            (head if ("lm_head" in name) else trunk).append(p)
        elif p.ndim == 1 and "norm" in name.lower():
            norms.append(p)
    common = dict(role="muon", weight_decay=0.0, momentum=momentum, nesterov=True,
                  ns_steps=ns_steps, rms_match=True, prox_rho=0.0, max_flip_rate=None,
                  state_dtype=state_dtype, ns_dtype=ns_dtype)
    groups = []
    if trunk:
        groups.append(dict(params=trunk, lr=master_lr, tag="trunk", **common))
    if head:
        groups.append(dict(params=head, lr=master_lr * head_lr_scale, tag="head", **common))
    if norms:
        if norm_lr is None:
            raise ValueError("trainable norms require norm_lr")
        groups.append(dict(
            params=norms,
            lr=norm_lr,
            tag="norms",
            role="adamw",
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
        ))
    return groups


# ---------------------------------------------------------------------------
# SignatureMeter: measure the run against the recovered fixed point
# ---------------------------------------------------------------------------

def _pack2(codes: Tensor) -> Tensor:
    """codes int8 in {-1,0,1} -> packed uint8, 4 codes/byte (CPU)."""
    c = (codes.reshape(-1) + 1).to(torch.uint8).cpu()
    pad = (-c.numel()) % 4
    if pad:
        c = torch.cat([c, torch.zeros(pad, dtype=torch.uint8)])
    c = c.reshape(-1, 4)
    return c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)


def _unpack2(packed: Tensor, numel: int) -> Tensor:
    out = torch.empty(packed.numel(), 4, dtype=torch.uint8)
    out[:, 0] = packed & 3
    out[:, 1] = (packed >> 2) & 3
    out[:, 2] = (packed >> 4) & 3
    out[:, 3] = (packed >> 6) & 3
    return out.reshape(-1)[:numel].to(torch.int8) - 1


@dataclass
class _ModuleRef:
    name: str
    module: AbsmeanTernaryLinear
    init_codes_packed: Tensor           # CPU uint8
    base_master: Optional[Tensor]       # CPU bf16 (movement/sign sample only)
    layer_idx: int


class SignatureMeter:
    """Tracks movement / churn / zeros / sign-inheritance vs the Prism targets.

    init snapshot = absmean-PTQ codes of the initial masters (packed 2-bit on
    CPU, ~0.27 bytes/param). Movement & sign inheritance are measured on a
    sampled subset of modules whose base masters are kept in bf16 on CPU.
    """

    def __init__(self, model: nn.Module, *, base_sample_frac: float = 0.25, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.refs: list[_ModuleRef] = []
        for name, module in model.named_modules():
            if not isinstance(module, AbsmeanTernaryLinear):
                continue
            codes, _ = module.codes_and_scales()
            keep_base = rng.random() < base_sample_frac
            m = re.search(r"layers\.(\d+)\.", name)
            self.refs.append(_ModuleRef(
                name=name,
                module=module,
                init_codes_packed=_pack2(codes),
                base_master=module.weight_master.detach().to(torch.bfloat16).cpu() if keep_base else None,
                layer_idx=int(m.group(1)) if m else -1,
            ))

    @torch.no_grad()
    def measure(self) -> dict[str, Any]:
        tot = zeros = churn = 0
        nz = sign_ok = 0
        mv_num = mv_den = 0.0
        per_layer: dict[int, list[float]] = {}
        for ref in self.refs:
            codes, scales = ref.module.codes_and_scales()
            n = codes.numel()
            init = _unpack2(ref.init_codes_packed, n).to(codes.device).reshape_as(codes)
            tot += n
            zeros += int((codes == 0).sum())
            churn += int((codes != init).sum())
            if ref.base_master is not None:
                base = ref.base_master.to(codes.device).float()
                deq = codes.float() * scales.unsqueeze(-1)
                deq = deq.reshape(base.shape)
                d2 = float((deq - base).square().sum())
                b2 = float(base.square().sum())
                mv_num += d2
                mv_den += b2
                nzm = codes != 0
                nz += int(nzm.sum())
                sign_ok += int((torch.sign(base.reshape_as(codes))[nzm] == codes[nzm].float()).sum())
                per_layer.setdefault(ref.layer_idx, []).append(math.sqrt(d2 / max(b2, 1e-12)))
        sig = {
            "zeros": zeros / max(tot, 1),
            "churn": churn / max(tot, 1),
            "movement": math.sqrt(mv_num / max(mv_den, 1e-12)),
            "sign_inherit": sign_ok / max(nz, 1),
            "per_layer_movement": {k: sum(v) / len(v) for k, v in sorted(per_layer.items())},
        }
        sig["distance"] = self.distance(sig)
        sig["in_band"] = all(
            SIGNATURE_BAND[k][0] <= sig[k] <= SIGNATURE_BAND[k][1] for k in SIGNATURE_BAND
        )
        return sig

    @staticmethod
    def distance(sig: dict[str, Any]) -> float:
        d = 0.0
        for k, target in SIGNATURE_TARGET.items():
            lo, hi = SIGNATURE_BAND[k]
            d += ((sig[k] - target) / max(hi - lo, 1e-6)) ** 2
        return math.sqrt(d / len(SIGNATURE_TARGET))


def movement_reference(u: float) -> float:
    """Expected dequant movement at budget fraction u (sqrt-diffusion growth
    of the learned component on top of the PTQ quantization floor)."""
    u = min(max(u, 0.0), 1.0)
    return math.sqrt(PTQ_MOVEMENT_FLOOR ** 2 + (LEARNED_MOVEMENT_AT_BUDGET ** 2) * u)


def churn_reference(u: float) -> float:
    """Expected support/code migration at budget fraction u.

    Movement is dominated by the PTQ floor early in training, so it is a poor
    controller signal. Code churn starts near zero and directly measures
    whether the low-bit support is migrating toward the Prism/Bonsai fixed
    point.
    """
    u = min(max(u, 0.0), 1.0)
    return SIGNATURE_TARGET["churn"] * math.sqrt(u)


# ---------------------------------------------------------------------------
# MovementServo: LR as a controlled variable
# ---------------------------------------------------------------------------

class MovementServo:
    """PI controller in log-LR space tracking churn_reference(u).

    Guardrails (forensic invariants, applied after the PI term):
      sign_inherit < 0.88  -> x0.7   (updates violent enough to break base signs)
      churn > 0.40         -> x0.8   (support churning past the fixed point)
    """

    def __init__(self, *, kp: float = 2.0, ki: float = 0.3,
                 min_mult: float = 0.1, max_mult: float = 3.0) -> None:
        self.kp, self.ki = kp, ki
        self.min_mult, self.max_mult = min_mult, max_mult
        self.integral = 0.0
        self.lr_mult = 1.0
        self.last_ref = 0.0
        self.last_err = 0.0
        self.signal = "churn"

    def update(self, sig: dict[str, Any], u: float) -> float:
        ref = churn_reference(u)
        err = ref - sig["churn"]
        self.last_ref = ref
        self.last_err = err
        self.integral = max(min(self.integral + err, 0.5), -0.5)
        mult = self.lr_mult * math.exp(self.kp * err + self.ki * self.integral)
        if sig["sign_inherit"] < SIGNATURE_BAND["sign_inherit"][0]:
            mult *= 0.7
        if sig["churn"] > 0.40:
            mult *= 0.8
        self.lr_mult = max(min(mult, self.max_mult), self.min_mult)
        return self.lr_mult

    def apply(self, optimizer: torch.optim.Optimizer, base_lrs: Sequence[float]) -> None:
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * self.lr_mult


# ---------------------------------------------------------------------------
# KLBanditMixer: data mix as a measured quantity
# ---------------------------------------------------------------------------

class KLBanditMixer:
    """Sampling probs over capability arms ∝ softmax(EMA per-arm KL / tau)."""

    def __init__(self, arms: Sequence[str], *, tau: float = 0.5, floor: float = 0.05,
                 ema: float = 0.8, seed: int = 0) -> None:
        self.arms = list(arms)
        self.tau, self.floor, self.ema = tau, floor, ema
        self.kl: dict[str, float] = {a: 1.0 for a in self.arms}
        self._rng = random.Random(seed)

    def update(self, arm: str, kl_value: float) -> None:
        self.kl[arm] = self.ema * self.kl[arm] + (1 - self.ema) * float(kl_value)

    def probs(self) -> dict[str, float]:
        mx = max(self.kl.values())
        exps = {a: math.exp((self.kl[a] - mx) / self.tau) for a in self.arms}
        z = sum(exps.values())
        n = len(self.arms)
        return {a: self.floor / n + (1 - self.floor) * exps[a] / z for a in self.arms}

    def sample(self) -> str:
        p = self.probs()
        return self._rng.choices(self.arms, weights=[p[a] for a in self.arms], k=1)[0]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def teacher_topk(teacher_logits: Tensor, *, k: int = 64) -> tuple[Tensor, Tensor]:
    """Extract the cacheable teacher side of the KD loss (raw top-k values+indices).

    Frozen probe sets should call this once at startup and reuse the result —
    probe evaluation then costs a student-only forward.
    """
    tv, ti = torch.topk(teacher_logits.detach(), k, dim=-1)
    return tv.float(), ti


def topk_kd_from_cache(student_logits: Tensor, tv: Tensor, ti: Tensor, *,
                       temperature: float = 1.0,
                       teacher_lse: Optional[Tensor] = None) -> Tensor:
    """Teacher top-k KL, with an exact aggregate tail when ``teacher_lse`` is given."""
    tvt = tv / temperature
    s_sel = torch.gather(student_logits, -1, ti).float() / temperature
    if teacher_lse is not None:
        t_lp = tvt - teacher_lse.unsqueeze(-1)
        s_lse = torch.logsumexp(student_logits.float() / temperature, dim=-1)
        s_lp = s_sel - s_lse.unsqueeze(-1)
        pt = t_lp.exp()
        ps = s_lp.exp()
        pt_other = (1.0 - pt.sum(dim=-1)).clamp_min(1e-8)
        ps_other = (1.0 - ps.sum(dim=-1)).clamp_min(1e-8)
        kl = (pt * (t_lp - s_lp)).sum(dim=-1)
        kl = kl + pt_other * (pt_other.log() - ps_other.log())
        return kl.mean() * (temperature * temperature)
    t_lp = tvt - torch.logsumexp(tvt, dim=-1, keepdim=True)
    s_lp = s_sel - torch.logsumexp(s_sel, dim=-1, keepdim=True)
    return (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1).mean() * (temperature * temperature)


def topk_kd_loss(student_logits: Tensor, teacher_logits: Tensor, *, k: int = 64,
                 temperature: float = 1.0) -> Tensor:
    """KL(teacher || student) over the teacher's top-k, renormalized.

    Vocab-safe for 248k vocabularies; ≫99% of teacher mass at T=1 lives in the
    top-64. Gathers before any fp32 upcast so peak memory is O(k), not O(V) —
    on a 248k vocab a full-tensor .float() would transiently cost ~4 GB per
    4k-token micro-batch. Inputs [..., V]; returns scalar mean over positions.
    """
    tv, ti = teacher_topk(teacher_logits, k=k)
    teacher_lse = torch.logsumexp(teacher_logits.detach().float() / temperature, dim=-1)
    return topk_kd_from_cache(
        student_logits, tv, ti, temperature=temperature, teacher_lse=teacher_lse
    )


def chunked_ce_loss(logits: Tensor, targets: Tensor, *, chunk: int = 512) -> Tensor:
    """Next-token CE computed in sequence chunks to bound fp32 logit memory."""
    lg = logits[:, :-1]
    tg = targets[:, 1:]
    total = lg.shape[0] * lg.shape[1]
    loss = lg.new_zeros((), dtype=torch.float32)
    for i in range(0, lg.shape[1], chunk):
        sl = lg[:, i : i + chunk].flatten(0, 1).float()
        st = tg[:, i : i + chunk].flatten()
        loss = loss + F.cross_entropy(sl, st, reduction="sum")
    return loss / total


def anchor_layer_indices(layer_types: Sequence[str]) -> list[int]:
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


def anchored_hidden_loss(student_hidden: Sequence[Tensor], teacher_hidden: Sequence[Tensor],
                         anchors: Sequence[int]) -> Tensor:
    """1 - cos on the residual stream after each full-attention layer.

    hidden_states[i+1] is the output of layer i. Anneal the weight of this
    loss to zero over the first ~20% of budget; it is a stabilizer for the
    DeltaNet recurrence, not part of the evidence-fixed recipe.
    """
    losses = []
    for i in anchors:
        hs = student_hidden[i + 1].float()
        ht = teacher_hidden[i + 1].float().detach()
        cos = F.cosine_similarity(hs.flatten(0, -2), ht.flatten(0, -2), dim=-1)
        losses.append(1.0 - cos.mean())
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Chunked-head losses: never materialize [seq, 248320] logits.
# Teacher: streaming top-k merge over vocab chunks (no_grad).
# Student: per-chunk gather + running logsumexp under torch.utils.checkpoint,
# so backward recomputes one chunk at a time (~130 MB peak instead of ~2 GB).
# ---------------------------------------------------------------------------

def head_weight(head_module: nn.Module) -> Tensor:
    """Resolve the effective output-projection weight, honoring ternary STE."""
    if isinstance(head_module, AbsmeanTernaryLinear):
        return head_module.quantized_weight(torch.bfloat16)
    return head_module.weight


@torch.no_grad()
def teacher_topk_chunked(h: Tensor, head_module: nn.Module, *, k: int = 64,
                         chunk: int = 16384, temperature: float = 1.0,
                         return_lse: bool = False):
    """Exact global top-k and optional full-vocabulary log normalizer."""
    W = head_weight(head_module)
    V = W.shape[0]
    tv = ti = None
    lse = torch.full(h.shape[:-1], torch.finfo(torch.float32).min, device=h.device)
    for lo in range(0, V, chunk):
        Wc = W[lo : lo + chunk]
        lg = h @ Wc.T.to(h.dtype)
        lse = torch.logaddexp(lse, torch.logsumexp(lg.float() / temperature, dim=-1))
        v, i = torch.topk(lg, min(k, lg.shape[-1]), dim=-1)
        i = i + lo
        if tv is None:
            tv, ti = v, i
        else:
            vv = torch.cat([tv, v], dim=-1)
            ii = torch.cat([ti, i], dim=-1)
            tv, pos = torch.topk(vv, k, dim=-1)
            ti = torch.gather(ii, -1, pos)
    if return_lse:
        return tv.float(), ti, lse
    return tv.float(), ti


def student_kd_ce_chunked(h: Tensor, head_module: nn.Module, tv: Tensor, ti: Tensor,
                          ids: Tensor, *, ce_weight: float = 0.0, chunk: int = 16384,
                          temperature: float = 1.0,
                          teacher_lse: Optional[Tensor] = None,
                          return_metrics: bool = False):
    """Top-K-plus-tail KL (+ optional CE) from hidden states, vocab-chunked.

    Numerically identical to topk_kd_loss(logits, ...) + chunked_ce_loss but
    with peak logit memory of ONE chunk; each chunk is checkpointed so backward
    recomputes its matmul instead of storing it.
    """
    from torch.utils.checkpoint import checkpoint

    W = head_weight(head_module)
    V = W.shape[0]
    B, S, K = ti.shape
    neg_inf = torch.finfo(torch.float32).min
    # next-token targets aligned to prediction positions (last position is a
    # dummy self-copy, masked out of the CE mean below)
    tgt = torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)

    def chunk_fn(h_in: Tensor, Wc: Tensor, lo: int):
        lg = (h_in @ Wc.T.to(h_in.dtype)).float()
        c = lg.shape[-1]
        lse_c = torch.logsumexp(lg, dim=-1)
        lse_t_c = torch.logsumexp(lg / temperature, dim=-1)
        max_c, max_i_c = torch.max(lg, dim=-1)
        in_c = (ti >= lo) & (ti < lo + c)
        sel = torch.where(in_c, ti - lo, torch.zeros_like(ti))
        g = torch.gather(lg, -1, sel)
        s_sel_c = torch.where(in_c, g, torch.full_like(g, neg_inf))
        tgt_in = (tgt >= lo) & (tgt < lo + c)
        tsel = torch.where(tgt_in, tgt - lo, torch.zeros_like(tgt))
        tg = torch.gather(lg, -1, tsel.unsqueeze(-1)).squeeze(-1)
        true_c = torch.where(tgt_in, tg, torch.full_like(tg, neg_inf))
        return s_sel_c, lse_c, lse_t_c, true_c, max_c, max_i_c

    s_sel = torch.full((B, S, K), neg_inf, device=h.device)
    lse = torch.full((B, S), neg_inf, device=h.device)
    lse_t = torch.full((B, S), neg_inf, device=h.device)
    true_lg = torch.full((B, S), neg_inf, device=h.device)
    max_lg = torch.full((B, S), neg_inf, device=h.device)
    max_idx = torch.zeros((B, S), dtype=torch.long, device=h.device)
    for lo in range(0, V, chunk):
        Wc = W[lo : lo + chunk]
        if torch.is_grad_enabled() and (h.requires_grad or Wc.requires_grad):
            s_c, l_c, lt_c, t_c, m_c, mi_c = checkpoint(
                chunk_fn, h, Wc, lo, use_reentrant=False
            )
        else:
            s_c, l_c, lt_c, t_c, m_c, mi_c = chunk_fn(h, Wc, lo)
        s_sel = torch.maximum(s_sel, s_c)   # each entry lives in exactly one chunk
        lse = torch.logaddexp(lse, l_c)
        lse_t = torch.logaddexp(lse_t, lt_c)
        true_lg = torch.maximum(true_lg, t_c)
        better = m_c > max_lg
        max_lg = torch.where(better, m_c, max_lg)
        max_idx = torch.where(better, mi_c + lo, max_idx)

    tvt = tv / temperature
    s_t = s_sel / temperature
    if teacher_lse is None:
        t_lp = tvt - torch.logsumexp(tvt, dim=-1, keepdim=True)
        s_lp = s_t - torch.logsumexp(s_t, dim=-1, keepdim=True)
        loss = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1).mean()
    else:
        t_lp = tvt - teacher_lse.unsqueeze(-1)
        s_lp = s_t - lse_t.unsqueeze(-1)
        pt = t_lp.exp()
        ps = s_lp.exp()
        pt_other = (1.0 - pt.sum(dim=-1)).clamp_min(1e-8)
        ps_other = (1.0 - ps.sum(dim=-1)).clamp_min(1e-8)
        loss = (pt * (t_lp - s_lp)).sum(dim=-1)
        loss = loss + pt_other * (pt_other.log() - ps_other.log())
        loss = loss.mean()
    loss = loss * (temperature * temperature)
    if ce_weight > 0:
        ce = (lse[:, :-1] - true_lg[:, :-1]).mean()
        loss = loss + ce_weight * ce
    if not return_metrics:
        return loss
    metrics = {
        "top1_agreement": (max_idx == ti[..., 0]).float().mean(),
        "teacher_topk_mass": (tvt - teacher_lse.unsqueeze(-1)).exp().sum(dim=-1).mean()
        if teacher_lse is not None else torch.ones((), device=h.device),
        "student_topk_mass": (s_t - lse_t.unsqueeze(-1)).exp().sum(dim=-1).mean(),
    }
    return loss, metrics


def hidden_forward(model: nn.Module, ids: Tensor, *, need_hidden: bool = False):
    """Run the text backbone WITHOUT the lm_head; return (last_hidden, hidden_states).

    Works for CausalLM (model.model) and ConditionalGeneration
    (model.model.language_model) layouts.
    """
    core = getattr(model, "model", model)
    base = getattr(core, "language_model", core)
    out = base(input_ids=ids, output_hidden_states=need_hidden, use_cache=False)
    return out.last_hidden_state, (out.hidden_states if need_hidden else None)


# ---------------------------------------------------------------------------
# BonsaiMuonFast: shape-batched Newton-Schulz
# ---------------------------------------------------------------------------

def _batched_ns(x: Tensor, *, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz orthogonalization on a stack [B, m, n] — one bmm chain
    instead of B sequential 2D loops. Same math as the scalar version."""
    norm = x.flatten(1).norm(dim=1).clamp_min(eps).view(-1, 1, 1)
    x = x / norm
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2)
    for _ in range(int(steps)):
        a = x @ x.transpose(1, 2)
        x = 1.5 * x - 0.5 * (a @ x)
    if transposed:
        x = x.transpose(1, 2)
    return x


class BonsaiMuonFast(torch.optim.Optimizer):
    """BonsaiMuon's muon path with Newton-Schulz batched across same-shape
    masters (e.g. 32 down_proj at once). Supports exactly the falsified-free
    recipe: momentum + nesterov + NS + rms_match. No prox, no flip region.
    Same-shaped 2D masters use batched Muon; small continuous compensation
    groups (for example RMSNorm weights) use the AdamW path below."""

    BNS = 8  # max matrices per batched-NS launch (bounds transient memory)

    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = dict(lr=1e-4, role="muon", momentum=0.95, nesterov=True,
                            ns_steps=5, rms_match=True, weight_decay=0.0,
                            state_dtype=torch.bfloat16, ns_dtype=torch.bfloat16)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        from collections import defaultdict

        for group in self.param_groups:
            role = group.get("role", "muon")
            if role == "adamw":
                for p in group["params"]:
                    if p.grad is not None:
                        self._adamw_step(p, group)
                continue
            if role != "muon":
                raise ValueError(f"unknown optimizer role: {role}")
            lr = float(group["lr"])
            momentum = float(group.get("momentum", 0.95))
            nesterov = bool(group.get("nesterov", True))
            ns_steps = int(group.get("ns_steps", 5))
            rms_match = bool(group.get("rms_match", True))
            sdt = group.get("state_dtype", torch.bfloat16)
            ndt = group.get("ns_dtype", torch.bfloat16)
            by_shape: dict[tuple, list] = defaultdict(list)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("BonsaiMuonFast expects 2D masters only")
                by_shape[tuple(p.shape)].append(p)
            for shape, plist in by_shape.items():
                for i in range(0, len(plist), self.BNS):
                    batch = plist[i : i + self.BNS]
                    raws = []
                    for p in batch:
                        st = self.state[p]
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(p, dtype=sdt)
                        buf = st["momentum_buffer"]
                        g = p.grad.detach()
                        buf.mul_(momentum).add_(g.to(buf.dtype))
                        raw = g.float() + momentum * buf.float() if nesterov else buf.float()
                        raws.append(raw)
                    U = torch.stack(raws)
                    X = _batched_ns(U.to(ndt), steps=ns_steps).float()
                    if rms_match:
                        raw_rms = U.square().mean(dim=(1, 2)).sqrt()
                        upd_rms = X.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-12)
                        X = X * (raw_rms / upd_rms).view(-1, 1, 1)
                    for p, x in zip(batch, X):
                        p.add_(x.to(p.dtype), alpha=-lr)
        return loss

    @torch.no_grad()
    def _adamw_step(self, p: Tensor, group: dict) -> None:
        grad = p.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("AdamW compensation path does not support sparse gradients")
        lr = float(group.get("lr", 1e-4))
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-8))
        wd = float(group.get("weight_decay", 0.0))
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] += 1
        if wd:
            p.mul_(1.0 - lr * wd)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        g = grad.to(p.dtype)
        exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        bc1 = 1.0 - beta1 ** state["step"]
        bc2 = 1.0 - beta2 ** state["step"]
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bc2)).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-(lr / bc1))


# ---------------------------------------------------------------------------
# Stopping rule + export
# ---------------------------------------------------------------------------

def should_stop(kl_history: Sequence[tuple[float, float]], sig: dict[str, Any], *,
                budget_frac: float, plateau_window: float = 0.15,
                plateau_improvement: float = 0.02) -> tuple[bool, str]:
    """(tokens_frac, probe_kl) history + current signature -> (stop?, reason)."""
    if budget_frac < 0.25 or len(kl_history) < 4:
        return False, "warmup"
    recent = [kl for u, kl in kl_history if u >= budget_frac - plateau_window]
    if len(recent) < 2:
        return False, "insufficient probes in window"
    improvement = (recent[0] - recent[-1]) / max(recent[0], 1e-9)
    if improvement >= plateau_improvement:
        return False, f"KL still falling ({improvement:.1%} over window)"
    if not sig.get("in_band", False):
        return False, f"KL plateaued but signature out of band (distance {sig['distance']:.2f})"
    return True, "KL plateau and signature in band"


@torch.no_grad()
def export_unpacked(model: nn.Module, out_dir: Path, *, shard_bytes: int = 10 * 2 ** 30,
                    original_names: Optional[Callable[[str], str]] = None) -> None:
    """Write Prism-convention 'unpacked' safetensors: ternary-dequant bf16 for
    wrapped modules, absmean-PTQ for the (frozen) embedding, frozen tensors
    as-is. Feed the result to the existing Clark pack pipeline."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Tensor] = {}
    wrapped_prefixes = []
    for name, module in model.named_modules():
        if isinstance(module, AbsmeanTernaryLinear):
            wrapped_prefixes.append(name)
            codes, scales = module.codes_and_scales()
            deq = (codes.float() * scales.unsqueeze(-1)).reshape(module.out_features, module.in_features)
            state[f"{name}.weight"] = deq.to(torch.bfloat16)
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias.detach().to(torch.bfloat16)
    for pname, p in model.named_parameters():
        owner = pname.rsplit(".", 1)[0]
        if any(owner == w or owner.startswith(w + ".") for w in wrapped_prefixes):
            continue
        t = p.detach()
        if pname.endswith("embed_tokens.weight"):
            g = t.float().reshape(-1, GROUP)
            s = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            already = float((torch.round(g / s).clamp_(-1, 1) * s - g).abs().max()) == 0.0
            if not already:  # PTQ here only if surgery didn't already do it
                codes, scales = absmean_quantize(t.float())
                t = (codes.float() * scales.unsqueeze(-1)).reshape(t.shape)
        state[pname if original_names is None else original_names(pname)] = t.to(torch.bfloat16)
    # simple size-based sharding + index
    shards: list[dict[str, Tensor]] = [{}]
    sizes = [0]
    for k in sorted(state):
        nbytes = state[k].numel() * state[k].element_size()
        if sizes[-1] + nbytes > shard_bytes and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = state[k]
        sizes[-1] += nbytes
    weight_map = {}
    n = len(shards)
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(out_dir / fname))
        for k in shard:
            weight_map[k] = fname
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map}, indent=1)
    )
