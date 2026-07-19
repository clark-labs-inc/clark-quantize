"""Quality-first Sana replay QAT.

Fresh FP16 -> optimal ternary master-weight QAT, supervised by FP16 teacher
replay states plus low-noise x0/image losses. This is intentionally not a
continuation of the all-g112 champion; that checkpoint is the control.
"""
import json, os, random, signal, sys, tempfile, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SanaPipeline
from PIL import Image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize import bonsai_muon as BM
from clark_quantize.sana_ckpt import load_compact_checkpoint, validate_quant_names

try:
    import lpips as lpips_lib
except Exception:
    lpips_lib = None


dev = "cuda"
MODEL = os.environ.get("MODEL", "models/Sana_1600M_512px")
POOL = os.environ.get("POOL", "sana_calib_256.jsonl")
HARDFILE = os.environ.get("HARDFILE", "shared_hard.json")
TAG = os.environ.get("TAG", "sana_replay_q1_g112")
K = int(os.environ.get("K", "20"))
G = int(os.environ.get("G", "112"))
GUID = float(os.environ.get("GUID", "4.5"))
STEPS = int(os.environ.get("STEPS", "3000"))
LATENTS = int(os.environ.get("LATENTS", "1"))
MAX_CALIB = int(os.environ.get("MAX_CALIB", "256"))
MAX_VAL = int(os.environ.get("MAX_VAL", "32"))
RENOISE_STATES = int(os.environ.get("RENOISE_STATES", "3"))
ALPHA_RAMP_STEPS = int(os.environ.get("ALPHA_RAMP_STEPS", "1000"))
RENDER_EVERY = int(os.environ.get("RENDER_EVERY", "500"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", str(RENDER_EVERY)))
FULL_CKPT_EVERY = int(os.environ.get("FULL_CKPT_EVERY", str(CKPT_EVERY)))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "1"))
SAVE_FULL_STEPS = int(os.environ.get("SAVE_FULL_STEPS", "0"))
SAVE_INITIAL_STATE = int(os.environ.get("SAVE_INITIAL_STATE", "1"))
DRY_RUN = int(os.environ.get("DRY_RUN", "0"))
STOP_FILE = os.environ.get("STOP_FILE", "")
RESUME = os.environ.get("RESUME", "")
OPT = os.environ.get("OPT", "adamw8bit").lower()

TRUNK_LR = float(os.environ.get("TRUNK_LR", "4e-5"))
SCALE_LR = float(os.environ.get("SCALE_LR", "1e-4"))
NORM_LR = float(os.environ.get("NORM_LR", "2e-4"))
COND_LR = float(os.environ.get("COND_LR", "2e-5"))
PATCH_LR = float(os.environ.get("PATCH_LR", "2e-6"))
SCALE_SHIFT_LR = float(os.environ.get("SCALE_SHIFT_LR", "0.0"))
CONV_DEPTH_LR = float(os.environ.get("CONV_DEPTH_LR", "0.0"))

LAM_VEL = float(os.environ.get("LAM_VEL", "1.0"))
LAM_X0 = float(os.environ.get("LAM_X0", "0.75"))
LAM_X0_EARLY = float(os.environ.get("LAM_X0_EARLY", str(LAM_X0)))
LAM_X0_RAMP_EXTRA_STEPS = int(os.environ.get("LAM_X0_RAMP_EXTRA_STEPS", "0"))
LAM_TRAJ = float(os.environ.get("LAM_TRAJ", "0.25"))
LAM_FEATURE = float(os.environ.get("LAM_FEATURE", "0.10"))
LAM_LPIPS = float(os.environ.get("LAM_LPIPS", "0.03"))
LAM_EDGE = float(os.environ.get("LAM_EDGE", "0.07"))
LAM_PATCH = float(os.environ.get("LAM_PATCH", "0.03"))
LAM_PIX = float(os.environ.get("LAM_PIX", "0.02"))
PERCEPTUAL_START_STEP = int(os.environ.get("PERCEPTUAL_START_STEP", "0"))
PERCEPTUAL_RAMP_STEPS = int(os.environ.get("PERCEPTUAL_RAMP_STEPS", str(max(1, ALPHA_RAMP_STEPS))))
CHAMPION_CKPT = os.environ.get("CHAMPION_CKPT", "")
LAM_CHAMP_V = float(os.environ.get("LAM_CHAMP_V", "0.0"))
CHAMPION_ANCHOR_START_STEP = int(os.environ.get("CHAMPION_ANCHOR_START_STEP", "0"))
CHAMPION_ANCHOR_RAMP_STEPS = int(os.environ.get("CHAMPION_ANCHOR_RAMP_STEPS", "250"))
CHAMPION_LOW_NOISE_ONLY = int(os.environ.get("CHAMPION_LOW_NOISE_ONLY", "1"))
PERCEPTUAL_EVERY = int(os.environ.get("PERCEPTUAL_EVERY", "8"))
EVAL_LPIPS_EVERY = int(os.environ.get("EVAL_LPIPS_EVERY", "100"))
BPTT_X0 = int(os.environ.get("BPTT_X0", "4"))
BPTT_M8 = int(os.environ.get("BPTT_M8", "4"))
M8 = int(os.environ.get("M8", "8"))
REFRESH = int(os.environ.get("REFRESH", "200"))
FORENSICS_EVERY = int(os.environ.get("FORENSICS_EVERY", "250"))
NHARD = int(os.environ.get("NHARD", "4"))
PROX_RHO_OVERRIDE = os.environ.get("PROX_RHO", "")

FULL_STATE_FORMAT = "sana_replay_qat_training_state_v1"
COMPACT_FORMAT = "sana_replay_qat_quant_artifact_v1"
shutdown = {"signum": None}


def log(*args):
    print(*args, flush=True)


def handle_signal(signum, frame):
    shutdown["signum"] = signum
    log(f"[signal] received {signum}; checkpointing at next safe point")


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def atomic_torch_save(payload, path):
    root = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=root)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json_save(payload, path):
    root = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=root, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def is_trunk_lin(n, m):
    return (
        isinstance(m, nn.Linear)
        and n.startswith("transformer_blocks")
        and (".attn1." in n or ".attn2." in n)
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))
    )


def is_trunk_conv(n, m):
    return isinstance(m, nn.Conv2d) and n.startswith("transformer_blocks") and (
        ".ff.conv_inverted" in n or ".ff.conv_point" in n
    )


def predicate(n, m):
    return is_trunk_lin(n, m) or is_trunk_conv(n, m)


PROTECTED_KEYS = (
    "patch_embed",
    "time_embed",
    "caption_projection",
    "caption_norm",
    "scale_shift_table",
    "norm_out",
    "proj_out",
    "conv_depth",
)


def island_lr(n):
    if "proj_out" in n:
        return 0.0
    if "caption_norm" in n or "norm_out" in n:
        return NORM_LR
    if "time_embed" in n or "caption_projection" in n:
        return COND_LR
    if "patch_embed" in n:
        return PATCH_LR
    if "scale_shift_table" in n:
        return SCALE_SHIFT_LR
    if "conv_depth" in n:
        return CONV_DEPTH_LR
    return 0.0


def is_protected_island(n):
    return any(k in n for k in PROTECTED_KEYS)


def choose_replay_k(rng):
    r = rng.random()
    if r < 0.20:
        return rng.randrange(0, max(1, min(4, K - 1)))
    if r < 0.50:
        return rng.randrange(4, max(5, min(10, K - 1)))
    return rng.randrange(10, max(11, K - 3))


def bucket(k):
    return "E" if k < 4 else ("M" if k < 10 else "L")


def pack_rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(0).cpu() if torch.cuda.is_available() else None,
    }


def restore_rng_state(st):
    if not st:
        return
    random.setstate(st["python"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and st.get("cuda") is not None:
        torch.cuda.set_rng_state(st["cuda"], device=0)


def optimizer_state_to_device(optimizer):
    for p, st in optimizer.state.items():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(p.device)


def run_config():
    return {
        "tag": TAG,
        "model": MODEL,
        "pool": POOL,
        "steps": STEPS,
        "latents": LATENTS,
        "max_calib": MAX_CALIB,
        "max_val": MAX_VAL,
        "renoise_states": RENOISE_STATES,
        "k": K,
        "g": G,
        "guidance": GUID,
        "alpha_ramp_steps": ALPHA_RAMP_STEPS,
        "optimizer": OPT,
        "trunk_lr": TRUNK_LR,
        "scale_lr": SCALE_LR,
        "norm_lr": NORM_LR,
        "cond_lr": COND_LR,
        "patch_lr": PATCH_LR,
        "conv_depth_lr": CONV_DEPTH_LR,
        "loss": {
            "vel": LAM_VEL,
            "x0": LAM_X0,
            "x0_early": LAM_X0_EARLY,
            "x0_ramp_extra_steps": LAM_X0_RAMP_EXTRA_STEPS,
            "traj": LAM_TRAJ,
            "feature": LAM_FEATURE,
            "lpips": LAM_LPIPS,
            "edge": LAM_EDGE,
            "patch": LAM_PATCH,
            "pix": LAM_PIX,
            "perceptual_start_step": PERCEPTUAL_START_STEP,
            "perceptual_ramp_steps": PERCEPTUAL_RAMP_STEPS,
            "champion_ckpt": CHAMPION_CKPT,
            "champion_v": LAM_CHAMP_V,
            "champion_anchor_start_step": CHAMPION_ANCHOR_START_STEP,
            "champion_anchor_ramp_steps": CHAMPION_ANCHOR_RAMP_STEPS,
            "champion_low_noise_only": CHAMPION_LOW_NOISE_ONLY,
        },
        "prox_rho_override": PROX_RHO_OVERRIDE,
    }


log(f"[cfg] TAG={TAG} POOL={POOL} STEPS={STEPS} LATENTS={LATENTS} MAX_CALIB={MAX_CALIB} RENOISE={RENOISE_STATES}")
pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.to(dev)
tf = pipe.transformer
vae = pipe.vae
pipe.scheduler.set_timesteps(K, device=dev)
SIG = pipe.scheduler.sigmas.float().to(dev)
VSCALE = getattr(vae.config, "scaling_factor", 1.0) or 1.0

try:
    tf.enable_gradient_checkpointing()
except Exception as exc:
    log("[warn] no transformer gradient checkpointing", exc)
try:
    vae.enable_gradient_checkpointing()
except Exception as exc:
    log("[warn] no vae gradient checkpointing", exc)
for p in vae.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def vae_decode_ng(x):
    return vae.decode((x / VSCALE).to(vae.dtype), return_dict=False)[0].float().clamp(-1, 1)


def vae_decode(x):
    return vae.decode((x / VSCALE).to(vae.dtype), return_dict=False)[0].float().clamp(-1, 1)


pool = [json.loads(l) for l in open(POOL)]
calib_rows = [r for r in pool if r.get("split") == "calib"][:MAX_CALIB]
val_rows = [r for r in pool if r.get("split") == "val"][:MAX_VAL]
hard_prompts = json.load(open(HARDFILE))[:NHARD]

ehs_pool, traj, x0T, ti_meta = {}, {}, {}, {}
cur = [0]
tinfo = {"ti": 0}


def teacher_hook(m, args, kwargs, out):
    y = out[0] if isinstance(out, (tuple, list)) else out.sample
    ti = tinfo["ti"]
    if ti not in ehs_pool:
        ehs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        extra = {
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in kwargs.items()
            if k not in ("hidden_states", "encoder_hidden_states", "timestep")
        }
        ehs_pool[ti] = (ehs.detach(), extra)
    hs = kwargs.get("hidden_states", args[0] if args else None)
    ts = kwargs.get("timestep", args[2] if len(args) > 2 else None)
    traj[(ti, cur[0])] = {"hs": hs.detach(), "ts": ts.detach(), "vT": y.detach()}
    cur[0] += 1


hook = tf.register_forward_hook(teacher_hook, with_kwargs=True)
calib_tis, val_tis = [], []
with torch.no_grad():
    ti = 0
    for split, rows, sink in (("calib", calib_rows, calib_tis), ("val", val_rows, val_tis)):
        nlat = LATENTS if split == "calib" else 1
        for row in rows:
            for li in range(nlat):
                tinfo["ti"] = ti
                cur[0] = 0
                sink.append(ti)
                ti_meta[ti] = {
                    "split": split,
                    "prompt": row["prompt"],
                    "source": row.get("source", ""),
                    "category": row.get("category", ""),
                    "latent_index": li,
                }
                out = pipe(
                    prompt=row["prompt"],
                    num_inference_steps=K,
                    height=512,
                    width=512,
                    guidance_scale=GUID,
                    output_type="latent",
                    generator=torch.Generator(dev).manual_seed(ti * 7 + li),
                )
                x0T[ti] = out.images.detach()[:1].to(torch.bfloat16)
                ti += 1
hook.remove()
log(f"[teacher] trajectories={len(traj)} calib_tis={len(calib_tis)} val_tis={len(val_tis)}")


def tf_call_latent(x1, ti, k):
    ehs, extra = ehs_pool[ti]
    d = traj[(ti, k)]
    lat = torch.cat([x1, x1], 0).to(d["hs"].dtype)
    out = tf(hidden_states=lat, encoder_hidden_states=ehs, timestep=d["ts"], **extra)
    return out[0] if isinstance(out, (tuple, list)) else out.sample


renoise_bank = []
rng_bank = random.Random(17)
with torch.no_grad():
    for ti in calib_tis:
        for _ in range(RENOISE_STATES):
            k = choose_replay_k(rng_bank)
            sigma = SIG[k].to(x0T[ti].device, dtype=torch.float32)
            eps = torch.randn(x0T[ti].shape, device=dev, dtype=torch.float32, generator=torch.Generator(dev).manual_seed(ti * 1009 + k))
            x = (1.0 - sigma) * x0T[ti].float() + sigma * eps
            vT = tf_call_latent(x.to(torch.bfloat16), ti, k).detach()
            renoise_bank.append({"ti": ti, "k": k, "x": x.to(torch.bfloat16), "vT": vT})
log(f"[replay] renoise_states={len(renoise_bank)} low_noise={sum(1 for r in renoise_bank if r['k'] >= 10)}")

teacher_img = {ti: vae_decode_ng(x0T[ti].float()) for ti in calib_tis}
with torch.no_grad():
    refs = [
        pipe(prompt=p, num_inference_steps=K, height=512, width=512, guidance_scale=GUID, generator=torch.Generator(dev).manual_seed(9000 + j)).images[0]
        for j, p in enumerate(hard_prompts)
    ]

BM.replace_modules_with_bonsai_quant(tf, predicate, mode="ternary", group_size=G, init="mse_ternary", forward_blend_alpha=0.0)
for p in tf.parameters():
    p.requires_grad_(False)
ste_named_mods = [(n, m) for n, m in tf.named_modules() if isinstance(m, BM._GroupwiseSTEBase)]
ste_mods = [m for _, m in ste_named_mods]
for _, m in ste_named_mods:
    m.weight_master.requires_grad_(True)
    m.log_scales.requires_grad_(True)
island_items = [(n, p) for n, p in tf.named_parameters() if is_protected_island(n)]
island_train = []
for n, p in island_items:
    lr = island_lr(n)
    if lr > 0:
        p.requires_grad_(True)
        island_train.append((n, p, lr))
masters = [m.weight_master for m in ste_mods]
scales = [m.log_scales for m in ste_mods]

import bitsandbytes as bnb

groups = [{"params": masters, "lr": TRUNK_LR}, {"params": scales, "lr": SCALE_LR}]
for n, p, lr in island_train:
    groups.append({"params": [p], "lr": lr})
opt = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.95), weight_decay=0.0)
LP = lpips_lib.LPIPS(net=os.environ.get("LPIPS_NET", "vgg")).to(dev).eval() if lpips_lib else None
if LP is not None:
    for p in LP.parameters():
        p.requires_grad_(False)
trainable = masters + scales + [p for _, p, _ in island_train]
init_codesign = [torch.sign(m._codes_from_matrix_detached(m.weight_master).float()).cpu() for m in ste_mods]
init_islands = {n: p.detach().clone() for n, p, _ in island_train}
log(f"[quant] ste={len(ste_mods)} masters={len(masters)} scales={len(scales)} islands_train={len(island_train)} lpips={LP is not None} gpu={torch.cuda.memory_allocated()/1e9:.1f}GB")


@torch.no_grad()
def dequant_compact_weight(code, log_scales, in_features):
    out, groups, group = code.shape
    scales = torch.exp(log_scales.float()).unsqueeze(-1)
    return (code.float() * scales).reshape(out, groups * group)[:, :in_features]


@torch.no_grad()
def snapshot_quant_state():
    return {
        "mods": [(m.weight_master.detach().cpu().clone(), m.log_scales.detach().cpu().clone()) for m in ste_mods],
        "islands": {n: p.detach().cpu().clone() for n, p in island_items},
    }


@torch.no_grad()
def restore_quant_state(snap):
    for m, (wm, ls) in zip(ste_mods, snap["mods"]):
        m.weight_master.data.copy_(wm.to(m.weight_master.device, dtype=m.weight_master.dtype))
        m.log_scales.data.copy_(ls.to(m.log_scales.device, dtype=m.log_scales.dtype))
    for n, p in island_items:
        src = snap["islands"].get(n)
        if src is not None:
            p.data.copy_(src.to(p.device, dtype=p.dtype))


@torch.no_grad()
def load_compact_quant_state(path):
    ck = load_compact_checkpoint(path)
    validate_quant_names(ck, [n for n, _ in ste_named_mods])
    for (name, m), code, log_scales, shape in zip(ste_named_mods, ck["codes"], ck["log_scales"], ck["shapes"]):
        w = dequant_compact_weight(code, log_scales, int(shape[1]))
        if tuple(w.shape) != tuple(m.weight_master.shape):
            raise ValueError(f"champion weight shape mismatch for {name}: {tuple(w.shape)} != {tuple(m.weight_master.shape)}")
        m.weight_master.data.copy_(w.to(m.weight_master.device, dtype=m.weight_master.dtype))
        m.log_scales.data.copy_(log_scales.to(m.log_scales.device, dtype=m.log_scales.dtype))
    islands = ck.get("islands", {})
    if isinstance(islands, dict):
        for n, p in island_items:
            if n in islands:
                p.data.copy_(islands[n].to(p.device, dtype=p.dtype))


if CHAMPION_CKPT and LAM_CHAMP_V > 0:
    snap = snapshot_quant_state()
    load_compact_quant_state(CHAMPION_CKPT)
    BM.set_bonsai_forward_blend_alpha(tf, 1.0)
    with torch.no_grad():
        for i, r in enumerate(renoise_bank):
            r["vC"] = tf_call_latent(r["x"], r["ti"], r["k"]).detach().to(torch.bfloat16)
            if (i + 1) % 128 == 0:
                log(f"[champion-anchor] precomputed {i + 1}/{len(renoise_bank)}")
    restore_quant_state(snap)
    del snap
    torch.cuda.empty_cache()
    log(f"[champion-anchor] ckpt={CHAMPION_CKPT} targets={sum(1 for r in renoise_bank if 'vC' in r)}")


def load_resume(path):
    ck = torch_load_cpu(path)
    if ck.get("format") != FULL_STATE_FORMAT:
        raise ValueError("train_sana_replay_qat resumes only full state files")
    tf.load_state_dict(ck["transformer"], strict=False)
    if ck.get("optimizer") and ck.get("config", {}).get("optimizer", OPT) == OPT:
        opt.load_state_dict(ck["optimizer"])
        optimizer_state_to_device(opt)
    restore_rng_state(ck.get("rng"))
    log(f"[resume] loaded {path} step {ck.get('global_step', 0)}")
    return int(ck.get("global_step", 0))


BASE_STEP = load_resume(RESUME) if RESUME else 0


SOB = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32, device=dev).unsqueeze(1)


def lum(img):
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def edge(img):
    return F.conv2d(lum(img), SOB.to(img.dtype), padding=1)


def edge_loss(a, b):
    return (edge(a) - edge(b)).abs().mean()


def patch_loss(a, b, ksz=160, n=2):
    if LP is None:
        return a.new_zeros(())
    with torch.no_grad():
        e = edge(b).abs().sum(1, keepdim=True)
        pooled = F.avg_pool2d(e, ksz, stride=ksz // 2)
        idx = pooled.flatten().topk(min(n, pooled.numel())).indices
        nw = pooled.shape[-1]
        coords = [((i // nw) * (ksz // 2), (i % nw) * (ksz // 2)) for i in idx.tolist()]
    loss = a.new_zeros(())
    for y, x in coords:
        y = min(y, a.shape[-2] - ksz)
        x = min(x, a.shape[-1] - ksz)
        loss = loss + LP(a[:, :, y : y + ksz, x : x + ksz], b[:, :, y : y + ksz, x : x + ksz]).mean()
    return loss / max(1, len(coords))


def guided_v(out):
    vu, vc = out.float().chunk(2, 0)
    return vu + GUID * (vc - vu)


def roll_to_x0(x1, ti, k, grad_last):
    x = x1.float()
    n = K - k
    for j in range(1, n + 1):
        r = k + j - 1
        ctx = torch.enable_grad() if j > n - grad_last else torch.no_grad()
        with ctx:
            v = guided_v(tf_call_latent(x.to(torch.bfloat16), ti, r))
            x = x + (SIG[r + 1] - SIG[r]) * v
        if j <= n - grad_last:
            x = x.detach()
    return x


def roll_m8(ti, k, grad_last):
    x = traj[(ti, k)]["hs"][:1].float()
    for j in range(1, M8 + 1):
        r = k + j - 1
        ctx = torch.enable_grad() if j > M8 - grad_last else torch.no_grad()
        with ctx:
            v = guided_v(tf_call_latent(x.to(torch.bfloat16), ti, r))
            x = x + (SIG[r + 1] - SIG[r]) * v
        if j <= M8 - grad_last:
            x = x.detach()
    return x


def alpha_at(global_step):
    if ALPHA_RAMP_STEPS <= 0:
        return 1.0
    return min(1.0, global_step / float(ALPHA_RAMP_STEPS))


def ramp01(global_step, start_step, ramp_steps):
    if global_step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (global_step - start_step) / float(ramp_steps))


def x0_weight_at(global_step):
    if LAM_X0_RAMP_EXTRA_STEPS <= 0:
        return LAM_X0_EARLY if global_step < ALPHA_RAMP_STEPS else LAM_X0
    t = ramp01(global_step, ALPHA_RAMP_STEPS, LAM_X0_RAMP_EXTRA_STEPS)
    return LAM_X0_EARLY + t * (LAM_X0 - LAM_X0_EARLY)


def perceptual_weight_at(global_step):
    return ramp01(global_step, PERCEPTUAL_START_STEP, PERCEPTUAL_RAMP_STEPS)


def champion_weight_at(global_step):
    if not CHAMPION_CKPT or LAM_CHAMP_V <= 0:
        return 0.0
    return LAM_CHAMP_V * ramp01(global_step, CHAMPION_ANCHOR_START_STEP, CHAMPION_ANCHOR_RAMP_STEPS)


def rho_at(global_step, alpha):
    if PROX_RHO_OVERRIDE:
        return float(PROX_RHO_OVERRIDE)
    if alpha < 1.0:
        return 0.0
    if global_step < 3000:
        return 0.005
    if global_step < 7000:
        return 0.02
    return 0.05


@torch.no_grad()
def prox_pull(rho):
    if rho <= 0:
        return
    for m in ste_mods:
        m.weight_master.data.lerp_(m._quantized_matrix_detached(), rho)


@torch.no_grad()
def refresh_scales():
    for m in ste_mods:
        m.reset_quant_scales("mse_ternary")


@torch.no_grad()
def forensics():
    zf = sf = sa = 0.0
    n = 0
    scales_mean = []
    for i, m in enumerate(ste_mods):
        code = m._codes_from_matrix_detached(m.weight_master).float()
        zf += (code == 0).float().sum().item()
        n += code.numel()
        cs = torch.sign(code).cpu()
        both = (code != 0).cpu() & (init_codesign[i] != 0)
        sf += ((cs != init_codesign[i]) & both).float().sum().item()
        sa += both.float().sum().item()
        scales_mean.append(m._scales().mean().item())
    isl = {}
    for name, p, _ in island_train:
        key = "caption_norm" if "caption_norm" in name else ("conv_depth" if "conv_depth" in name else "other")
        isl[key] = isl.get(key, 0.0) + (p.detach() - init_islands[name]).norm().item()
    return zf / max(n, 1), sf / max(sa, 1), sum(scales_mean) / len(scales_mean), isl


@torch.no_grad()
def eval_val_lpips(limit=8):
    if LP is None or not val_tis:
        return 0.0
    total = 0.0
    for ti in val_tis[:limit]:
        x = roll_to_x0(traj[(ti, 12)]["hs"][:1], ti, 12, 0)
        total += LP(vae_decode_ng(x), vae_decode_ng(x0T[ti].float())).mean().item()
    return total / min(limit, len(val_tis))


@torch.no_grad()
def render(tag):
    old = [m.forward_blend_alpha for m in ste_mods]
    BM.set_bonsai_forward_blend_alpha(tf, 1.0)
    imgs = [
        pipe(prompt=p, num_inference_steps=K, height=512, width=512, guidance_scale=GUID, generator=torch.Generator(dev).manual_seed(9000 + j)).images[0]
        for j, p in enumerate(hard_prompts)
    ]
    for m, a in zip(ste_mods, old):
        m.forward_blend_alpha = a
    canvas = Image.new("RGB", (512 * len(hard_prompts), 1024), (20, 20, 20))
    for j in range(len(hard_prompts)):
        canvas.paste(refs[j], (j * 512, 0))
        canvas.paste(imgs[j], (j * 512, 512))
    path = f"render_{TAG}_{tag}.png"
    canvas.save(path)
    log(f"[render] {path}")


def compact_payload(global_step):
    return {
        "format": COMPACT_FORMAT,
        "codes": [m._codes_from_matrix_detached(m.weight_master).cpu() for m in ste_mods],
        "log_scales": [m.log_scales.detach().cpu().clone() for m in ste_mods],
        "shapes": [tuple(m.weight_master.shape) for m in ste_mods],
        "quant_names": [n for n, _ in ste_named_mods],
        "group_sizes": [m.group_size for m in ste_mods],
        "islands": {n: p.detach().cpu().clone() for n, p in island_items},
        "mode": "ternary",
        "group_size": G,
        "step": global_step,
        "config": run_config(),
    }


def save_ckpt(global_step, force_full=False):
    compact = compact_payload(global_step)
    atomic_torch_save(compact, f"ckpt_{TAG}.pt")
    compact_step = None
    if SAVE_STEPS:
        compact_step = f"ckpt_{TAG}_s{global_step}.pt"
        atomic_torch_save(compact, compact_step)
    full_due = force_full or (FULL_CKPT_EVERY > 0 and global_step % FULL_CKPT_EVERY == 0) or shutdown["signum"] is not None or global_step == BASE_STEP + STEPS
    full_latest = None
    full_step = None
    if full_due:
        full_latest = f"state_{TAG}.pt"
        full = {
            "format": FULL_STATE_FORMAT,
            "version": 1,
            "tag": TAG,
            "global_step": global_step,
            "config": run_config(),
            "quant": compact,
            "transformer": to_cpu(tf.state_dict()),
            "optimizer": to_cpu(opt.state_dict()),
            "rng": to_cpu(pack_rng_state()),
            "shutdown_signal": shutdown["signum"],
        }
        atomic_torch_save(full, full_latest)
        if SAVE_FULL_STEPS:
            full_step = f"state_{TAG}_s{global_step}.pt"
            atomic_torch_save(full, full_step)
    manifest = {
        "tag": TAG,
        "global_step": global_step,
        "compact_latest": f"ckpt_{TAG}.pt",
        "compact_step": compact_step,
        "full_latest": full_latest,
        "full_step": full_step,
        "config": run_config(),
        "shutdown_signal": shutdown["signum"],
    }
    atomic_json_save(manifest, f"manifest_{TAG}.json")
    log(f"[ckpt] step={global_step} compact=ckpt_{TAG}.pt{(' full=' + full_latest) if full_latest else ''}")


BM.set_bonsai_forward_blend_alpha(tf, alpha_at(BASE_STEP))
render(f"s{BASE_STEP}" if BASE_STEP else "s0")
if SAVE_INITIAL_STATE and BASE_STEP == 0:
    save_ckpt(0, force_full=True)
if DRY_RUN:
    log("[dry-run] initialized teacher bank, quantized model, and rendered s0")
    sys.exit(0)

calib_m8 = [(ti, k) for ti in calib_tis for k in range(0, K - M8)]
t0 = time.time()
last_val_lpips = eval_val_lpips()
for step in range(1, STEPS + 1):
    global_step = BASE_STEP + step
    alpha = alpha_at(global_step)
    rho = rho_at(global_step, alpha)
    BM.set_bonsai_forward_blend_alpha(tf, alpha)
    opt.zero_grad(set_to_none=True)

    q = random.choice(renoise_bank)
    out = tf_call_latent(q["x"], q["ti"], q["k"])
    Lv = F.mse_loss(out.float(), q["vT"].float())
    xS = roll_to_x0(q["x"], q["ti"], q["k"], BPTT_X0)
    xT = x0T[q["ti"]].float()
    Lx0 = F.mse_loss(xS, xT) / (xT.pow(2).mean() + 1e-6)

    ti_m8, k_m8 = random.choice(calib_m8)
    xm = roll_m8(ti_m8, k_m8, BPTT_M8)
    xt = traj[(ti_m8, k_m8 + M8)]["hs"][:1].float()
    Lm8 = F.mse_loss(xm, xt) / (xt.pow(2).mean() + 1e-6)
    Lfeat = F.l1_loss(guided_v(out), guided_v(q["vT"]))

    Lc = out.new_zeros(())
    champw = champion_weight_at(global_step)
    use_champion = champw > 0 and "vC" in q and (not CHAMPION_LOW_NOISE_ONLY or q["k"] >= 10)
    if use_champion:
        Lc = F.mse_loss(guided_v(out), guided_v(q["vC"]))

    Lp = Le = Lpa = Lpx = xS.new_zeros(())
    percw = perceptual_weight_at(global_step)
    do_perc = percw > 0 and PERCEPTUAL_EVERY > 0 and (global_step % PERCEPTUAL_EVERY == 0) and q["k"] >= 10
    if do_perc:
        IS = vae_decode(xS)
        IT = teacher_img[q["ti"]]
        if LP is not None:
            Lp = LP(IS, IT).mean()
            Lpa = patch_loss(IS, IT)
        Le = edge_loss(IS, IT)
        Lpx = (IS - IT).abs().mean()

    x0w = x0_weight_at(global_step)
    loss = LAM_VEL * Lv + x0w * Lx0 + LAM_TRAJ * Lm8 + LAM_FEATURE * Lfeat + champw * Lc
    loss = loss + percw * (LAM_LPIPS * Lp + LAM_EDGE * Le + LAM_PATCH * Lpa + LAM_PIX * Lpx)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 0.5)
    if torch.isfinite(gn):
        opt.step()
    prox_pull(rho)
    if REFRESH > 0 and global_step % REFRESH == 0:
        refresh_scales()

    if EVAL_LPIPS_EVERY > 0 and (global_step % EVAL_LPIPS_EVERY == 0 or step == 1):
        last_val_lpips = eval_val_lpips()
    if global_step % 25 == 0 or step == 1:
        log(
            f"[s{global_step}] a{alpha:.2f} rho{rho:.3f} k{q['k']} {bucket(q['k'])} "
            f"Lv{Lv.item():.2e} Lx0{Lx0.item():.2e} Lm8{Lm8.item():.2e} "
            f"Lc{Lc.item():.2e} x0w{x0w:.2f} cw{champw:.2f} pw{percw:.2f} "
            f"Lp{Lp.item():.2e} edge{Le.item():.2e} patch{Lpa.item():.2e} gn{float(gn):.2f} "
            f"valLPIPS{last_val_lpips:.4f} ({time.time() - t0:.0f}s)"
        )
    if global_step % FORENSICS_EVERY == 0:
        zf, sf, sm, isl = forensics()
        isls = " ".join(f"{k}{v:.2f}" for k, v in isl.items())
        log(f"[forensics s{global_step}] zero {zf:.3f} flip {sf:.4f} scale {sm:.3f} [{isls}]")
    if global_step % RENDER_EVERY == 0:
        render(f"s{global_step}")
    if global_step % CKPT_EVERY == 0:
        save_ckpt(global_step)
    if shutdown["signum"] is not None or (STOP_FILE and os.path.exists(STOP_FILE)):
        save_ckpt(global_step, force_full=True)
        log(f"[done] stopped at step {global_step}")
        break
else:
    render("final")
    save_ckpt(BASE_STEP + STEPS, force_full=True)
    log("[done]")
