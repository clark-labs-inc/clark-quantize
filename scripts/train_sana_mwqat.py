"""MW-QAT v2 — fresh master-weight QAT from FP16 Sana. Support is a DERIVED consequence of continuous
master-weight movement crossing the ternary threshold (the ONLY mechanism that produced coherent Sana
pure-ternary; explicit mask/code/soft-mask parameterizations all froze or moved support wrongly).
  W_master = W_fp16 ; W_fwd = Q(W_master) ; grad to master via STE (BM._GroupwiseSTEBase).
KEY change vs q1024t: DIFFERENTIATED island LR (caption_norm = Sana's final-norm compensator, HIGH LR;
proj_out FROZEN; embedders low; conv_depth stays FP16/frozen). Closed-form scale refresh every REFRESH
steps (reset_quant_scales) + trainable log_scales. NO density loss, NO scale prior, NO endpoint, NO mask
logits. alpha-ramp soft landing for optimization; alpha=1 hard renders are the ONLY quality verdict.
Proximal pull toward Q(master) on a schedule. Forensics every 500 (zero frac, sign flips, scale by depth,
proj_out delta MUST be 0, caption_norm delta should grow). 4-GPU DDP. Control: START_HARD=1 ALPHA_RAMP=0.
ENV: POOL=sana_calib_256.jsonl STEPS=6000 RAMP=1200 REFRESH=200 TAG=mwq1
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time, random, json, signal, tempfile
import torch.distributed as dist
os.environ["HF_HUB_OFFLINE"]="1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from diffusers import SanaPipeline
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize import bonsai_muon as BM
from clark_quantize.stein_qat import (
    flow_velocity_to_score,
    make_rademacher_projection,
    stein_imq_field,
    stein_transport_loss,
    tensor_rms,
)
try:
    import lpips as lpips_lib
except Exception:
    lpips_lib=None
WORLD=int(os.environ.get("WORLD_SIZE","1")); RANK=int(os.environ.get("RANK","0")); LRANK=int(os.environ.get("LOCAL_RANK","0")); DDP=WORLD>1
if DDP: dist.init_process_group("nccl"); torch.cuda.set_device(LRANK)
dev=f"cuda:{LRANK}"; MAIN=(RANK==0)
def log(*a):
    if MAIN: print(*a,flush=True)
MODEL="models/Sana_1600M_512px"; K=20; G=int(os.environ.get("G",os.environ.get("GSIZE","112"))); GUID=4.5; M8=8
MODE=os.environ.get("MODE","ternary"); STEPS=int(os.environ.get("STEPS","6000")); TAG=os.environ.get("TAG","mwq1")
POOL=os.environ.get("POOL","sana_calib_256.jsonl"); LATENTS=int(os.environ.get("LATENTS","2"))
RAMP=int(os.environ.get("RAMP","1200")); P2START=int(os.environ.get("P2START",str(RAMP))); KBPTD=int(os.environ.get("KBPTD","4"))
REFRESH=int(os.environ.get("REFRESH","200")); START_HARD=int(os.environ.get("START_HARD","0"))
PROX_RHO_OVERRIDE=os.environ.get("PROX_RHO","")
RENDER_EVERY=int(os.environ.get("RENDER_EVERY","500")); SAVE_STEPS=int(os.environ.get("SAVE_STEPS","0"))
CKPT_EVERY=int(os.environ.get("CKPT_EVERY",str(RENDER_EVERY if SAVE_STEPS else 1000)))
FORENSE=int(os.environ.get("FORENSICS_EVERY",os.environ.get("FORENSE","500")))
HEARTBEAT_EVERY=int(os.environ.get("HEARTBEAT_EVERY","0"))
RESUME=os.environ.get("RESUME","")
SAVE_FULL_STATE=int(os.environ.get("SAVE_FULL_STATE","1"))
FULL_CKPT_EVERY=int(os.environ.get("FULL_CKPT_EVERY",str(CKPT_EVERY)))
SAVE_INITIAL_STATE=int(os.environ.get("SAVE_INITIAL_STATE","1"))
STOP_FILE=os.environ.get("STOP_FILE","")
OPT=os.environ.get("OPT","adamw8bit").lower()
TRUNK_LR=float(os.environ.get("TRUNK_LR","2e-5")); SCALE_LR=float(os.environ.get("SCALE_LR","1e-4"))
TRAIN_SCOPE=os.environ.get("TRAIN_SCOPE","all").lower()
FREEZE_ATTENTION_STEPS=int(os.environ.get("FREEZE_ATTENTION_STEPS","0"))
LOAD_OPTIMIZER=int(os.environ.get("LOAD_OPTIMIZER","1"))
THRESHOLD_BOOST=float(os.environ.get("THRESHOLD_BOOST","0.0"))
THRESHOLD_TEMP=float(os.environ.get("THRESHOLD_TEMP","0.20"))
THRESHOLD_START=int(os.environ.get("THRESHOLD_START","0"))
THRESHOLD_RAMP=int(os.environ.get("THRESHOLD_RAMP","0"))
MAX_UPDATE_RMS=float(os.environ.get("MAX_UPDATE_RMS","0.0"))
QAT8_USE_MUON=int(os.environ.get("QAT8_USE_MUON","0"))
QAT8_MUON_EVERY=int(os.environ.get("QAT8_MUON_EVERY","16"))
QAT8_MUON_STEPS=int(os.environ.get("QAT8_MUON_STEPS","2"))
QAT8_MUON_MAX_ELEMENTS=int(os.environ.get("QAT8_MUON_MAX_ELEMENTS","8000000"))
QAT8_BLOCK_SIZE=int(os.environ.get("QAT8_BLOCK_SIZE","4096"))
NHARD=int(os.environ.get("NHARD","4")); HARDFILE=os.environ.get("HARDFILE","shared_hard.json")
ANCHOR_LAMBDA=float(os.environ.get("ANCHOR_LAMBDA","0.0"))
ANCHOR_CATEGORIES={c.strip().lower() for c in os.environ.get("ANCHOR_CATEGORIES","anchor").split(",") if c.strip()}
ANCHOR_MAX_STATES=int(os.environ.get("ANCHOR_MAX_STATES","0"))
CODEBOOK_LAMBDA=float(os.environ.get("CODEBOOK_LAMBDA","0.0"))
CODEBOOK_LAMBDA_START=float(os.environ.get("CODEBOOK_LAMBDA_START",str(CODEBOOK_LAMBDA)))
CODEBOOK_START=int(os.environ.get("CODEBOOK_START","0"))
CODEBOOK_RAMP=int(os.environ.get("CODEBOOK_RAMP","0"))
CODEBOOK_SAMPLE_MODULES=int(os.environ.get("CODEBOOK_SAMPLE_MODULES","0"))
MARGIN_LAMBDA=float(os.environ.get("MARGIN_LAMBDA","0.0"))
MARGIN_START=int(os.environ.get("MARGIN_START","0"))
MARGIN_RAMP=int(os.environ.get("MARGIN_RAMP","0"))
MARGIN_TEMP=float(os.environ.get("MARGIN_TEMP","0.15"))
IMAGE_LOSS_EVERY=int(os.environ.get("IMAGE_LOSS_EVERY","0"))
IMAGE_START=int(os.environ.get("IMAGE_START","500"))
IMAGE_RAMP=int(os.environ.get("IMAGE_RAMP","500"))
IMAGE_MIN_K=int(os.environ.get("IMAGE_MIN_K","12"))
IMAGE_BPTT=int(os.environ.get("IMAGE_BPTT","2"))
IMAGE_MAX_STATES=int(os.environ.get("IMAGE_MAX_STATES","0"))
LOWPASS_K=int(os.environ.get("LOWPASS_K","17"))
PATCH_SIZE=int(os.environ.get("PATCH_SIZE","160"))
PATCH_COUNT=int(os.environ.get("PATCH_COUNT","2"))
LAM_COMP=float(os.environ.get("LAM_COMP","0.0"))
LAM_DETAIL=float(os.environ.get("LAM_DETAIL","0.0"))
LAM_EDGE=float(os.environ.get("LAM_EDGE","0.0"))
LAM_PATCH=float(os.environ.get("LAM_PATCH","0.0"))
LAM_PIX=float(os.environ.get("LAM_PIX","0.0"))
COMP_CKPT=os.environ.get("COMP_CKPT","")
COMP_ANCHOR_MAX=int(os.environ.get("COMP_ANCHOR_MAX","64"))
BELL_ENABLE=int(os.environ.get("BELL_ENABLE","0"))
BELL_START=int(os.environ.get("BELL_START","0"))
BELL_RAMP=int(os.environ.get("BELL_RAMP","500"))
BELL_ACTOR_LAMBDA=float(os.environ.get("BELL_ACTOR_LAMBDA","0.0"))
BELL_ACTOR_EVERY=int(os.environ.get("BELL_ACTOR_EVERY","4"))
BELL_CRITIC_EVERY=int(os.environ.get("BELL_CRITIC_EVERY","1"))
BELL_CRITIC_UPDATES=int(os.environ.get("BELL_CRITIC_UPDATES","1"))
BELL_CRITIC_LR=float(os.environ.get("BELL_CRITIC_LR","1e-4"))
BELL_HIDDEN=int(os.environ.get("BELL_HIDDEN","512"))
BELL_GAMMA=float(os.environ.get("BELL_GAMMA","0.95"))
BELL_TARGET_CLAMP=float(os.environ.get("BELL_TARGET_CLAMP","2.0"))
BELL_TEACHER_WEIGHT=float(os.environ.get("BELL_TEACHER_WEIGHT","0.25"))
BELL_NOISE_WEIGHT=float(os.environ.get("BELL_NOISE_WEIGHT","0.10"))
BELL_NOISE_STD=float(os.environ.get("BELL_NOISE_STD","0.08"))
JEFFREYS_ENABLE=int(os.environ.get("JEFFREYS_ENABLE","0"))
JEFFREYS_EVERY=int(os.environ.get("JEFFREYS_EVERY","1"))
JEFFREYS_LAMBDA=float(os.environ.get("JEFFREYS_LAMBDA","0.25"))
JEFFREYS_HORIZON=int(os.environ.get("JEFFREYS_HORIZON","4"))
JEFFREYS_MIN_K=int(os.environ.get("JEFFREYS_MIN_K","4"))
STEIN_ENABLE=int(os.environ.get("STEIN_ENABLE","0"))
STEIN_START=int(os.environ.get("STEIN_START","100"))
STEIN_RAMP=int(os.environ.get("STEIN_RAMP","400"))
STEIN_EVERY=int(os.environ.get("STEIN_EVERY","4"))
STEIN_PARTICLES=int(os.environ.get("STEIN_PARTICLES","4"))
STEIN_HORIZON=int(os.environ.get("STEIN_HORIZON","4"))
STEIN_MIN_K=int(os.environ.get("STEIN_MIN_K","10"))
STEIN_PROJ_DIM=int(os.environ.get("STEIN_PROJ_DIM","128"))
STEIN_PROJ_SEED=int(os.environ.get("STEIN_PROJ_SEED","5151"))
STEIN_KERNEL_OFFSET=float(os.environ.get("STEIN_KERNEL_OFFSET","1.0"))
STEIN_BETA=float(os.environ.get("STEIN_BETA","-0.5"))
STEIN_TARGET_RATIO=float(os.environ.get("STEIN_TARGET_RATIO","0.05"))
STEIN_MAX_WEIGHT=float(os.environ.get("STEIN_MAX_WEIGHT","0.10"))
STEIN_NORMALIZE_FIELD=int(os.environ.get("STEIN_NORMALIZE_FIELD","1"))
PTD_ENABLE=int(os.environ.get("PTD_ENABLE","0"))
PTD_SANITY_ONLY=int(os.environ.get("PTD_SANITY_ONLY","0"))
PTD_PAIRS=os.environ.get("PTD_PAIRS","data/sana_ptd_pairs.json")
PTD_INTERACTIONS=os.environ.get("PTD_INTERACTIONS","data/sana_ptd_interactions.json")
PTD_EVERY=int(os.environ.get("PTD_EVERY","4"))
PTD_INT_EVERY=int(os.environ.get("PTD_INT_EVERY","8"))
PTD_LAMBDA=float(os.environ.get("PTD_LAMBDA","0.02"))
PTD_INT_LAMBDA=float(os.environ.get("PTD_INT_LAMBDA","0.01"))
PTD_TARGET_RATIO=float(os.environ.get("PTD_TARGET_RATIO","0.0"))
PTD_INT_TARGET_RATIO=float(os.environ.get("PTD_INT_TARGET_RATIO","0.0"))
PTD_MAX_WEIGHT=float(os.environ.get("PTD_MAX_WEIGHT","0.10"))
PTD_INT_MAX_WEIGHT=float(os.environ.get("PTD_INT_MAX_WEIGHT","0.05"))
PTD_MIN_K=int(os.environ.get("PTD_MIN_K","8"))
PTD_HORIZON=int(os.environ.get("PTD_HORIZON","2"))
PTD_SANITY_SAMPLES=int(os.environ.get("PTD_SANITY_SAMPLES","8"))
TEACHER_COPY=int(os.environ.get("TEACHER_COPY",str(1 if (JEFFREYS_ENABLE or STEIN_ENABLE or PTD_ENABLE or PTD_SANITY_ONLY) else 0)))
log(f"[cfg] MODE={MODE} OPT={OPT} POOL={POOL} STEPS={STEPS} RAMP={RAMP} REFRESH={REFRESH} START_HARD={START_HARD} TAG={TAG} RESUME={RESUME or '-'} | DDP world={WORLD} rank={RANK}")
if RESUME.lower()=="auto":
    for cand in (f"state_{TAG}.pt", f"ckpt_{TAG}.pt"):
        if os.path.exists(cand):
            RESUME=cand; break
    else:
        RESUME=""
    log(f"[resume] auto -> {RESUME or 'none'}")

FULL_STATE_FORMAT="sana_mwqat_training_state_v1"
COMPACT_FORMAT="sana_mwqat_quant_artifact_v2"
shutdown={"signum":None}
def _handle_signal(signum, frame):
    shutdown["signum"]=signum
    log(f"[signal] received {signum}; checkpointing at next safe point")
signal.signal(signal.SIGTERM,_handle_signal)
signal.signal(signal.SIGINT,_handle_signal)

def stop_requested():
    local=shutdown["signum"] is not None or (STOP_FILE and os.path.exists(STOP_FILE))
    if DDP:
        t=torch.tensor([1 if local else 0],device=dev,dtype=torch.int32)
        dist.all_reduce(t,op=dist.ReduceOp.MAX)
        local=bool(t.item())
        if local and shutdown["signum"] is None:
            shutdown["signum"]=0
    return bool(local)

def torch_load_cpu(path):
    try:
        return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        return torch.load(path,map_location="cpu")

def to_cpu(obj):
    if torch.is_tensor(obj): return obj.detach().cpu().clone()
    if isinstance(obj,dict): return {k:to_cpu(v) for k,v in obj.items()}
    if isinstance(obj,list): return [to_cpu(v) for v in obj]
    if isinstance(obj,tuple): return tuple(to_cpu(v) for v in obj)
    return obj

def atomic_torch_save(payload,path):
    d=os.path.dirname(os.path.abspath(path)) or "."
    fd,tmp=tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",suffix=".tmp",dir=d)
    os.close(fd)
    try:
        torch.save(payload,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def atomic_json_save(payload,path):
    d=os.path.dirname(os.path.abspath(path)) or "."
    fd,tmp=tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",suffix=".tmp",dir=d,text=True)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(payload,f,indent=2,sort_keys=True)
            f.write("\n")
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)   # conv_depth (3x3 depthwise) excluded -> stays FP16, frozen
def _blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1
CAPNORM_LR=float(os.environ.get("CAPNORM_LR","1e-4")); COND_LR=float(os.environ.get("COND_LR","1e-5")); PATCH_LR=float(os.environ.get("PATCH_LR","1e-6"))
def island_lr(n):   # Prism image policy: proj_out FROZEN; caption_norm = Sana final-norm compensator (norm_out is parameter-free here)
    if "proj_out" in n: return 0.0                                              # frozen (Prism: bit-identical to base)
    if "norm_out" in n: return 1e-4                                             # parameter-free SanaModulatedNorm -> no-op on Sana
    if "caption_norm" in n: return CAPNORM_LR                                   # routes the compensator LR here (norm_out absent)
    if any(k in n for k in ("time_embed","caption_projection")): return COND_LR
    if "patch_embed" in n: return PATCH_LR
    if "scale_shift_table" in n: return 0.0
    return 0.0
def is_island(n): return island_lr(n)>0 or "proj_out" in n
GFFN=int(os.environ.get("GFFN",str(G))); GLATE=int(os.environ.get("GLATE",str(G))); NLATE=int(os.environ.get("NLATE","5")); PER_MODULE_GROUP=int(os.environ.get("PER_MODULE_GROUP","0"))
def group_for(n):   # per-module group size for mixed-rate (run B): late blocks override FFN rates
    if _blk(n)>=20-NLATE: return GLATE
    if ".ff." in n: return GFFN
    return G
def train_master_for(n):
    if TRAIN_SCOPE in ("late_detail","late_ffn"):
        return _blk(n)>=20-NLATE and ".ff." in n
    return True
def train_scale_for(n):
    if TRAIN_SCOPE in ("late_detail","late_scales"):
        return _blk(n)>=20-NLATE
    if TRAIN_SCOPE=="late_ffn":
        return _blk(n)>=20-NLATE and ".ff." in n
    return True
def replace_per_module(model, pred, gfn, mode):   # AUDIT FIX: actually apply per-module group size (BM.replace hard-codes one group)
    _init="mse_ternary" if mode=="ternary" else "absmean"
    for name,module in list(model.named_modules()):
        if not pred(name,module): continue
        g=gfn(name)
        if isinstance(module,nn.Linear): new=BM.QuantizedLinearSTE.from_linear(module, mode=mode, group_size=g, init=_init, forward_blend_alpha=1.0)
        elif isinstance(module,nn.Conv2d): new=BM.QuantizedConv1x1STE.from_conv2d(module, mode=mode, group_size=g, init=_init, forward_blend_alpha=1.0)
        else: continue
        parent,child=BM._get_parent_and_child(model,name); setattr(parent,child,new.to(next(model.parameters()).device))

pool=[json.loads(l) for l in open(POOL)]
calib_rows=[r for r in pool if r["split"]=="calib"]; val_rows=[r for r in pool if r["split"]=="val"]
calib_rows=calib_rows[RANK::WORLD]; val_rows=val_rows[RANK::WORLD]
calibP=[r["prompt"] for r in calib_rows]; valP=[r["prompt"] for r in val_rows]
hardP=json.load(open(HARDFILE))[:NHARD]
def is_anchor_row(row):
    cat=str(row.get("category","")).lower(); src=str(row.get("source","")).lower()
    return bool(ANCHOR_CATEGORIES and (cat in ANCHOR_CATEGORIES or src in ANCHOR_CATEGORIES or "anchor" in cat or "anchor" in src))
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer; vae=pipe.vae
TRANSFORMER_GRAD_CKPT=int(os.environ.get("TRANSFORMER_GRAD_CKPT",str(1 if TEACHER_COPY else 0)))
if TRANSFORMER_GRAD_CKPT:
    try:
        tf.enable_gradient_checkpointing()
        log("[grad-ckpt] enabled transformer gradient checkpointing")
    except Exception as e:
        log("[warn] no transformer grad-ckpt",e)
pipe.scheduler.set_timesteps(K,device=dev); SIG=pipe.scheduler.sigmas.float().to(dev)
VSCALE=getattr(vae.config,"scaling_factor",1.0) or 1.0
try: vae.enable_gradient_checkpointing()
except Exception as e: log("[warn] no vae grad-ckpt",e)
for p in vae.parameters(): p.requires_grad_(False)
ehs_pool={}; traj={}; ti_meta={}; x0T={}; cur=[0]; tinfo={"ti":0}
def hook(m,a,k,o):
    out=o[0] if isinstance(o,(tuple,list)) else o.sample; ti=tinfo["ti"]
    if ti not in ehs_pool:
        ehs=k.get("encoder_hidden_states",a[1] if len(a)>1 else None)
        ehs_pool[ti]=(ehs.detach(),{kk:v.detach() if torch.is_tensor(v) else v for kk,v in k.items() if kk not in("hidden_states","encoder_hidden_states","timestep")})
    hs=k.get("hidden_states",a[0] if a else None); ts=k.get("timestep",a[2] if len(a)>2 else None)
    traj[(ti,cur[0])]={"hs":hs.detach(),"ts":ts.detach(),"vT":out.detach()}; cur[0]+=1
h=tf.register_forward_hook(hook,with_kwargs=True)
calib_tis=[]; val_tis=[]
with torch.no_grad():
    ti=0
    for split,rows,sink in (("calib",calib_rows,calib_tis),("val",val_rows,val_tis)):
        nlat=LATENTS if split=="calib" else 1
        for pi,row in enumerate(rows):
            p=row["prompt"]
            for s in range(nlat):
                tinfo["ti"]=ti; cur[0]=0; sink.append(ti)
                ti_meta[ti]={"split":split,"prompt":p,"source":row.get("source",""),
                             "category":row.get("category",""),"prompt_index":pi,
                             "latent_index":s,"anchor":is_anchor_row(row)}
                out=pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,output_type="latent",generator=torch.Generator(dev).manual_seed(ti*7+s))
                x0T[ti]=out.images.detach()[:1].to(torch.bfloat16); ti+=1
h.remove()
def bucket(k): return 'E' if k<4 else ('M' if k<8 else 'L')
calib_chunks=[(ti,k) for ti in calib_tis for k in range(K-M8)]; val_chunks=[(ti,k) for ti in val_tis for k in range(K-M8)]
calib_states=[(ti,k) for ti in calib_tis for k in range(K)]
cs_b={'E':[],'M':[],'L':[]};
for s in calib_states: cs_b[bucket(s[1])].append(s)
cc_b={'E':[],'M':[],'L':[]}
for c in calib_chunks: cc_b[bucket(c[1])].append(c)
EVALC=[val_chunks[i] for i in sorted(random.Random(0).sample(range(len(val_chunks)),min(24,len(val_chunks))))]
refs=[]
if MAIN:
    with torch.no_grad():
        refs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
log(f"[bank] rank{RANK} {len(traj)} states | calib_tis {len(calib_tis)} val_tis {len(val_tis)} | refs {len(refs)}")
teacher_tf=None
if TEACHER_COPY:
    log("[teacher] loading frozen FP16 teacher copy for student-state path/Stein")
    teacher_pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16)
    teacher_tf=teacher_pipe.transformer.to(dev).eval()
    teacher_pipe.transformer=None
    del teacher_pipe
    torch.cuda.empty_cache()
    for p in teacher_tf.parameters(): p.requires_grad_(False)
    log(f"[teacher] ready GPU {torch.cuda.memory_allocated(LRANK)/1e9:.1f}GB")

# ---- quantize (master-weight STE) ----
if PER_MODULE_GROUP:
    replace_per_module(tf, predicate, group_for, MODE); log(f"[quant] per-module groups: G{G} FFN{GFFN} late{GLATE}x{NLATE}")
else:
    BM.replace_modules_with_bonsai_quant(tf, predicate, mode=MODE, group_size=G, init=("mse_ternary" if MODE=="ternary" else "absmean"), forward_blend_alpha=1.0)
for p in tf.parameters(): p.requires_grad_(False)
ste_named_mods=[(n,m) for n,m in tf.named_modules() if isinstance(m,BM._GroupwiseSTEBase)]
ste_mods=[m for _,m in ste_named_mods]
attention_trainable=[]
for n,m in ste_named_mods:
    m.weight_master.requires_grad_(train_master_for(n))
    m.log_scales.requires_grad_(train_scale_for(n))
    if ".attn" in n:
        if m.weight_master.requires_grad: attention_trainable.append(m.weight_master)
        if m.log_scales.requires_grad: attention_trainable.append(m.log_scales)
island_params=[]
for n,p in tf.named_parameters():
    lr=island_lr(n)
    if lr>0: p.requires_grad_(True); island_params.append((n,p,lr))
def run_config():
    return {"mode":MODE,"tag":TAG,"pool":POOL,"latents":LATENTS,"steps":STEPS,"ramp":RAMP,
            "p2start":P2START,"kbptd":KBPTD,"refresh":REFRESH,"start_hard":START_HARD,
            "prox_rho_override":PROX_RHO_OVERRIDE,
            "render_every":RENDER_EVERY,"ckpt_every":CKPT_EVERY,"full_ckpt_every":FULL_CKPT_EVERY,
            "heartbeat_every":HEARTBEAT_EVERY,
            "save_steps":SAVE_STEPS,"save_full_state":SAVE_FULL_STATE,
            "save_initial_state":SAVE_INITIAL_STATE,"stop_file":STOP_FILE,"g":G,"gffn":GFFN,
            "glate":GLATE,"nlate":NLATE,"per_module_group":PER_MODULE_GROUP,
            "optimizer":OPT,"load_optimizer":LOAD_OPTIMIZER,
            "trunk_lr":TRUNK_LR,"scale_lr":SCALE_LR,
            "train_scope":TRAIN_SCOPE,"freeze_attention_steps":FREEZE_ATTENTION_STEPS,
            "threshold_boost":THRESHOLD_BOOST,"threshold_temp":THRESHOLD_TEMP,
            "threshold_start":THRESHOLD_START,"threshold_ramp":THRESHOLD_RAMP,
            "anchor_lambda":ANCHOR_LAMBDA,"anchor_categories":sorted(ANCHOR_CATEGORIES),
            "anchor_max_states":ANCHOR_MAX_STATES,
            "codebook_lambda":CODEBOOK_LAMBDA,
            "codebook_lambda_start":CODEBOOK_LAMBDA_START,
            "codebook_start":CODEBOOK_START,
            "codebook_ramp":CODEBOOK_RAMP,
            "codebook_sample_modules":CODEBOOK_SAMPLE_MODULES,
            "margin_lambda":MARGIN_LAMBDA,
            "margin_start":MARGIN_START,
            "margin_ramp":MARGIN_RAMP,
            "margin_temp":MARGIN_TEMP,
            "image_loss_every":IMAGE_LOSS_EVERY,"image_start":IMAGE_START,
            "image_ramp":IMAGE_RAMP,"image_min_k":IMAGE_MIN_K,
            "image_bptt":IMAGE_BPTT,"image_max_states":IMAGE_MAX_STATES,
            "lowpass_k":LOWPASS_K,"patch_size":PATCH_SIZE,"patch_count":PATCH_COUNT,
            "lam_comp":LAM_COMP,"lam_detail":LAM_DETAIL,
            "lam_edge":LAM_EDGE,"lam_patch":LAM_PATCH,"lam_pix":LAM_PIX,
            "comp_ckpt":COMP_CKPT,"comp_anchor_max":COMP_ANCHOR_MAX,
            "bell_enable":BELL_ENABLE,"bell_start":BELL_START,
            "bell_ramp":BELL_RAMP,"bell_actor_lambda":BELL_ACTOR_LAMBDA,
            "bell_actor_every":BELL_ACTOR_EVERY,
            "bell_critic_every":BELL_CRITIC_EVERY,
            "bell_critic_updates":BELL_CRITIC_UPDATES,
            "bell_critic_lr":BELL_CRITIC_LR,
            "bell_hidden":BELL_HIDDEN,"bell_gamma":BELL_GAMMA,
            "bell_target_clamp":BELL_TARGET_CLAMP,
            "bell_teacher_weight":BELL_TEACHER_WEIGHT,
            "bell_noise_weight":BELL_NOISE_WEIGHT,
            "bell_noise_std":BELL_NOISE_STD,
            "jeffreys_enable":JEFFREYS_ENABLE,
            "jeffreys_every":JEFFREYS_EVERY,
            "jeffreys_lambda":JEFFREYS_LAMBDA,
            "jeffreys_horizon":JEFFREYS_HORIZON,
            "jeffreys_min_k":JEFFREYS_MIN_K,
            "stein_enable":STEIN_ENABLE,
            "stein_start":STEIN_START,
            "stein_ramp":STEIN_RAMP,
            "stein_every":STEIN_EVERY,
            "stein_particles":STEIN_PARTICLES,
            "stein_horizon":STEIN_HORIZON,
            "stein_min_k":STEIN_MIN_K,
            "stein_proj_dim":STEIN_PROJ_DIM,
            "stein_proj_seed":STEIN_PROJ_SEED,
            "stein_kernel_offset":STEIN_KERNEL_OFFSET,
            "stein_beta":STEIN_BETA,
            "stein_target_ratio":STEIN_TARGET_RATIO,
            "stein_max_weight":STEIN_MAX_WEIGHT,
            "stein_normalize_field":STEIN_NORMALIZE_FIELD,
            "ptd_enable":PTD_ENABLE,
            "ptd_sanity_only":PTD_SANITY_ONLY,
            "ptd_pairs":PTD_PAIRS,
            "ptd_interactions":PTD_INTERACTIONS,
            "ptd_every":PTD_EVERY,
            "ptd_int_every":PTD_INT_EVERY,
            "ptd_lambda":PTD_LAMBDA,
            "ptd_int_lambda":PTD_INT_LAMBDA,
            "ptd_target_ratio":PTD_TARGET_RATIO,
            "ptd_int_target_ratio":PTD_INT_TARGET_RATIO,
            "ptd_max_weight":PTD_MAX_WEIGHT,
            "ptd_int_max_weight":PTD_INT_MAX_WEIGHT,
            "ptd_min_k":PTD_MIN_K,
            "ptd_horizon":PTD_HORIZON,
            "ptd_sanity_samples":PTD_SANITY_SAMPLES,
            "teacher_copy":TEACHER_COPY,
            "max_update_rms":MAX_UPDATE_RMS,"qat8_use_muon":QAT8_USE_MUON,
            "qat8_muon_every":QAT8_MUON_EVERY,"qat8_muon_steps":QAT8_MUON_STEPS,
            "qat8_muon_max_elements":QAT8_MUON_MAX_ELEMENTS,"qat8_block_size":QAT8_BLOCK_SIZE,
            "capnorm_lr":CAPNORM_LR,"cond_lr":COND_LR,"patch_lr":PATCH_LR,
            "world":WORLD,"model":MODEL,"k":K,"guidance":GUID,"m8":M8}

def compact_payload(global_step):
    return {"format":COMPACT_FORMAT,
            "codes":[m._codes_from_matrix_detached(m.weight_master).cpu() for m in ste_mods],
            "log_scales":[m.log_scales.detach().cpu().clone() for m in ste_mods],
            "shapes":[tuple(m.weight_master.shape) for m in ste_mods],
            "quant_names":[n for n,_ in ste_named_mods],
            "group_sizes":[getattr(m,"group_size",None) for m in ste_mods],
            "islands":{n:p.detach().cpu().clone() for n,p,_ in island_params},
            "mode":MODE,"group_size":G,"step":global_step,"config":run_config()}

def load_compact_model(ck,path):
    if ck.get("mode") != MODE: raise ValueError(f"resume mode {ck.get('mode')} != {MODE}")
    if len(ck["codes"]) != len(ste_mods): raise ValueError(f"resume trunk count {len(ck['codes'])} != {len(ste_mods)}")
    names=ck.get("quant_names")
    if names and names != [n for n,_ in ste_named_mods]:
        raise ValueError("resume quant module order does not match current code/config")
    scale_migrated=0
    for m,code,ls,shp in zip(ste_mods,ck["codes"],ck["log_scales"],ck["shapes"]):
        code=code.float().to(dev); ls=ls.float().to(dev)
        out,ng,g=code.shape; in_features=int(shp[1])
        q=(code*ls.exp().unsqueeze(-1)).reshape(out,ng*g)[:,:in_features]
        if tuple(q.shape) != tuple(m.weight_master.shape): raise ValueError(f"resume shape {tuple(q.shape)} != {tuple(m.weight_master.shape)}")
        m.weight_master.data.copy_(q.to(m.weight_master.device,dtype=m.weight_master.dtype))
        if tuple(ls.shape) == tuple(m.log_scales.shape):
            m.log_scales.data.copy_(ls.to(m.log_scales.device,dtype=m.log_scales.dtype))
        else:
            m.reset_quant_scales("mse_ternary" if MODE=="ternary" else "absmean")
            scale_migrated+=1
    islands=ck.get("islands",{})
    if isinstance(islands,dict):
        missing=[]
        for n,p,_ in island_params:
            src=islands.get(n)
            if src is None: missing.append(n); continue
            p.data.copy_(src.to(p.device,dtype=p.dtype))
        if missing: log(f"[resume] warning missing {len(missing)} island tensors")
    extra=f" scale_migrated={scale_migrated}" if scale_migrated else ""
    log(f"[resume] loaded compact {path} step {ck.get('step',0)}{extra}")
    return int(ck.get("step",0))

def reset_migrated_scales(skipped):
    skipped=set(skipped); reset=0
    for n,m in ste_named_mods:
        if f"{n}.log_scales" in skipped:
            m.reset_quant_scales("mse_ternary" if MODE=="ternary" else "absmean")
            reset+=1
    return reset

@torch.no_grad()
def load_resume(path):
    ck=torch_load_cpu(path)
    if ck.get("format")==FULL_STATE_FORMAT:
        state=ck.get("transformer")
        if not state: raise ValueError(f"full resume {path} is missing transformer state")
        try:
            missing,unexpected=tf.load_state_dict(state,strict=False)
            if missing or unexpected:
                raise ValueError(f"full resume state mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
            log(f"[resume] loaded full model {path} step {ck.get('global_step',ck.get('step',0))}")
            return int(ck.get("global_step",ck.get("step",0))), ck
        except RuntimeError as e:
            cur=tf.state_dict(); compatible={}; skipped={}; unexpected=[]
            for k,v in state.items():
                if k not in cur:
                    unexpected.append(k); continue
                if torch.is_tensor(v) and torch.is_tensor(cur[k]) and tuple(v.shape)==tuple(cur[k].shape):
                    compatible[k]=v
                else:
                    skipped[k]=(tuple(v.shape),tuple(cur[k].shape))
            non_scale=[k for k in skipped if not k.endswith(".log_scales")]
            if unexpected or non_scale:
                raise ValueError(f"full resume incompatible unexpected={unexpected[:5]} non_scale={non_scale[:5]}") from e
            missing,unexpected2=tf.load_state_dict(compatible,strict=False)
            extra_missing=[k for k in missing if k not in skipped]
            if extra_missing or unexpected2:
                raise ValueError(f"full resume filtered mismatch missing={extra_missing[:5]} unexpected={unexpected2[:5]}") from e
            reset=reset_migrated_scales(skipped)
            migrated=dict(ck); migrated["optimizer"]=None; migrated["rank_rng"]=None
            log(f"[resume] migrated full model {path} step {ck.get('global_step',ck.get('step',0))}: reset {reset} resized scale tensors, skipped optimizer")
            return int(ck.get("global_step",ck.get("step",0))), migrated
    return load_compact_model(ck,path), None
BASE_STEP,RESUME_STATE=load_resume(RESUME) if RESUME else (0,None)
masters=[m.weight_master for m in ste_mods if m.weight_master.requires_grad]
scales=[m.log_scales for m in ste_mods if m.log_scales.requires_grad]
if OPT in ("bonsai_qat8","qat8","bonsaiqat8"):
    from clark_quantize.bonsai_qat8 import BonsaiQAT8
    groups=[]
    if masters:
        groups.append({"params":masters,"role":"matrix_qat8","lr":TRUNK_LR,"weight_decay":0.0,
                       "betas":(0.9,0.95),"eps":1e-8,"block_size":QAT8_BLOCK_SIZE,
                       "threshold_boost":0.0,"threshold_temp":THRESHOLD_TEMP,
                       "max_update_rms":MAX_UPDATE_RMS,"use_muon":bool(QAT8_USE_MUON),
                       "muon_every":QAT8_MUON_EVERY,"muon_steps":QAT8_MUON_STEPS,
                       "muon_max_elements":QAT8_MUON_MAX_ELEMENTS,"muon_rms_match":True,
                       "prox_rho":0.0})
    if scales:
        groups.append({"params":scales,"role":"adamw","lr":SCALE_LR,"weight_decay":0.0,
                       "betas":(0.9,0.95),"eps":1e-8})
    for n,p,lr in island_params:
        groups.append({"params":[p],"role":"adamw","lr":lr,"weight_decay":0.0,
                       "betas":(0.9,0.95),"eps":1e-8})
    opt=BonsaiQAT8(groups)
    OPT_KIND="bonsai_qat8"
else:
    import bitsandbytes as bnb
    groups=[]
    if masters: groups.append({"params":masters,"lr":TRUNK_LR})
    if scales: groups.append({"params":scales,"lr":SCALE_LR})
    for n,p,lr in island_params: groups.append({"params":[p],"lr":lr})
    opt=bnb.optim.AdamW8bit(groups,betas=(0.9,0.95),weight_decay=0.0)
    OPT_KIND="adamw8bit"
trainable=masters+scales+[p for _,p,_ in island_params]
log(f"[trainable] scope={TRAIN_SCOPE} masters={len(masters)} scales={len(scales)} islands={len(island_params)} attention_freeze_params={len(attention_trainable)}")
def optimizer_state_to_device(optimizer):
    for p,st in optimizer.state.items():
        for k,v in list(st.items()):
            if torch.is_tensor(v):
                st[k]=v.to(p.device)

def pack_rng_state():
    return {"rank":RANK,
            "python":random.getstate(),
            "torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state(LRANK).cpu() if torch.cuda.is_available() else None}

def collect_rank_rng_states():
    local=pack_rng_state()
    if not DDP: return [local]
    gathered=[None for _ in range(WORLD)]
    dist.all_gather_object(gathered,local)
    return gathered

def restore_rng_state(rank_rng):
    if not rank_rng: return
    st=None
    for cand in rank_rng:
        if cand and cand.get("rank")==RANK:
            st=cand; break
    if st is None: st=rank_rng[min(RANK,len(rank_rng)-1)] or rank_rng[0]
    random.setstate(st["python"])
    torch.set_rng_state(st["torch"])
    if torch.cuda.is_available() and st.get("cuda") is not None:
        torch.cuda.set_rng_state(st["cuda"],device=LRANK)
def freeze_attention_grads(branch_step):
    if FREEZE_ATTENTION_STEPS<=0 or branch_step>FREEZE_ATTENTION_STEPS:
        return
    for p in attention_trainable:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

def resume_optimizer_name(state):
    cfg=state.get("config",{}) if isinstance(state,dict) else {}
    return cfg.get("optimizer","adamw8bit")

if RESUME_STATE is not None and RESUME_STATE.get("optimizer"):
    prev_opt=resume_optimizer_name(RESUME_STATE)
    if LOAD_OPTIMIZER and prev_opt==OPT_KIND:
        opt.load_state_dict(RESUME_STATE["optimizer"])
        optimizer_state_to_device(opt)
        log(f"[resume] loaded optimizer state from {RESUME}")
    else:
        log(f"[resume] skipped optimizer state prev={prev_opt} current={OPT_KIND} load={LOAD_OPTIMIZER}")
if RESUME_STATE is not None and RESUME_STATE.get("rank_rng"):
    restore_rng_state(RESUME_STATE["rank_rng"])
    log(f"[resume] restored rank RNG state from {RESUME}")

def threshold_boost_at(global_step):
    if THRESHOLD_BOOST<=0 or global_step<THRESHOLD_START:
        return 0.0
    if THRESHOLD_RAMP<=0:
        return THRESHOLD_BOOST
    return THRESHOLD_BOOST*min(1.0,max(0.0,(global_step-THRESHOLD_START)/float(THRESHOLD_RAMP)))
def ramped_value(branch_step,start,ramp,lo,hi):
    if hi<=0 or branch_step<start:
        return 0.0
    if ramp<=0:
        return hi
    t=min(1.0,max(0.0,(branch_step-start)/float(ramp)))
    return lo+t*(hi-lo)
def codebook_lambda_at(branch_step):
    return ramped_value(branch_step,CODEBOOK_START,CODEBOOK_RAMP,CODEBOOK_LAMBDA_START,CODEBOOK_LAMBDA)
def margin_lambda_at(branch_step):
    return ramped_value(branch_step,MARGIN_START,MARGIN_RAMP,MARGIN_LAMBDA,MARGIN_LAMBDA)
@torch.no_grad()
def prox_pull(rho):
    if rho<=0: return
    for m in ste_mods: m.weight_master.data.lerp_(m._quantized_matrix_detached(), rho)
@torch.no_grad()
def refresh_scales():
    for m in ste_mods: m.reset_quant_scales("mse_ternary" if MODE=="ternary" else "absmean")
# forensics snapshots (init = PTQ / FP16)
init_codesign=[torch.sign(m._codes_from_matrix_detached(m.weight_master).float()).cpu() for m in ste_mods]
init_island={n:p.detach().float().norm().item() for n,p,_ in island_params}
init_island_val={n:p.detach().clone() for n,p,_ in island_params}
LP=None
if lpips_lib is not None and LAM_PATCH>0 and IMAGE_LOSS_EVERY>0:
    LP=lpips_lib.LPIPS(net=os.environ.get("LPIPS_NET","vgg")).to(dev).eval()
    for p in LP.parameters(): p.requires_grad_(False)
log(f"[mwqat] {len(ste_mods)} STE trunk | islands {len(island_params)} | GPU {torch.cuda.memory_allocated(LRANK)/1e9:.1f}GB")

def tf_call(hs,ehs,ts,ex):
    o=tf(hidden_states=hs,encoder_hidden_states=ehs,timestep=ts,**ex); return (o[0] if isinstance(o,(tuple,list)) else o.sample)
def teacher_tf_call(hs,ehs,ts,ex):
    if teacher_tf is None:
        raise RuntimeError("teacher copy is required for Jeffreys/Stein objectives")
    dtype=next(teacher_tf.parameters()).dtype
    with torch.no_grad():
        o=teacher_tf(hidden_states=hs.to(dtype),encoder_hidden_states=ehs.to(dtype),timestep=ts,**ex)
    return (o[0] if isinstance(o,(tuple,list)) else o.sample)
def guided_student_velocity(x,ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
    o=tf_call(lat,ehs,d["ts"],ex).float(); vu,vc=o.chunk(2,0)
    return vu+GUID*(vc-vu)
@torch.no_grad()
def guided_teacher_velocity(x,ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
    o=teacher_tf_call(lat,ehs,d["ts"],ex).float(); vu,vc=o.chunk(2,0)
    return vu+GUID*(vc-vu)
def load_json_list(path, required):
    if not path:
        return []
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return []
    rows=json.load(open(path))
    if not isinstance(rows,list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows
def capture_prompt_condition(prompt):
    captured={}
    def cond_hook(m,a,k,o):
        if captured:
            return
        ehs=k.get("encoder_hidden_states",a[1] if len(a)>1 else None)
        ex={kk:(v.detach() if torch.is_tensor(v) else v)
            for kk,v in k.items()
            if kk not in ("hidden_states","encoder_hidden_states","timestep")}
        captured["ehs"]=ehs.detach()
        captured["ex"]=ex
    h=tf.register_forward_hook(cond_hook,with_kwargs=True)
    try:
        with torch.no_grad():
            pipe(
                prompt=prompt,
                num_inference_steps=1,
                height=512,
                width=512,
                guidance_scale=GUID,
                output_type="latent",
                generator=torch.Generator(dev).manual_seed(12345),
            )
    finally:
        h.remove()
    if "ehs" not in captured:
        raise RuntimeError(f"failed to capture PTD condition for prompt: {prompt}")
    return captured
def ptd_make_bank():
    pair_rows=load_json_list(PTD_PAIRS, bool(PTD_ENABLE or PTD_SANITY_ONLY))
    int_rows=load_json_list(PTD_INTERACTIONS, False)
    prompts=[]
    for r in pair_rows:
        prompts.extend([r["positive"],r["negative"]])
    for r in int_rows:
        prompts.extend([r["ab"],r["a"],r["b"],r["null"]])
    bank={}
    for p in sorted(set(prompts)):
        bank[p]=capture_prompt_condition(p)
    pairs=[dict(r,pos=bank[r["positive"]],neg=bank[r["negative"]]) for r in pair_rows]
    interactions=[dict(r,ab_c=bank[r["ab"]],a_c=bank[r["a"]],b_c=bank[r["b"]],null_c=bank[r["null"]]) for r in int_rows]
    if MAIN and (PTD_ENABLE or PTD_SANITY_ONLY):
        log(f"[ptd] pairs={len(pairs)} interactions={len(interactions)} conditions={len(bank)}")
    return pairs,interactions
ptd_pairs,ptd_interactions=([],[])
if PTD_ENABLE or PTD_SANITY_ONLY:
    ptd_pairs,ptd_interactions=ptd_make_bank()
def ptd_state():
    horizon=max(0,min(PTD_HORIZON,K-2))
    max_k=max(0,K-horizon-1)
    min_k=min(max(0,PTD_MIN_K),max_k)
    ti=random.choice(calib_tis)
    k=random.randint(min_k,max_k) if max_k>=min_k else 0
    x=traj[(ti,k)]["hs"][:1].float().detach()
    with torch.no_grad():
        for j in range(horizon):
            r=k+j
            vS=guided_student_velocity(x,ti,r)
            x=x+(SIG[r+1]-SIG[r])*vS
    return x.detach(),traj[(ti,k+horizon)]["ts"]
def guided_student_condition(x,ts,cond):
    lat=torch.cat([x,x],0).to(next(tf.parameters()).dtype)
    o=tf_call(lat,cond["ehs"],ts,cond["ex"]).float()
    vu,vc=o.chunk(2,0)
    return vu+GUID*(vc-vu)
@torch.no_grad()
def guided_teacher_condition(x,ts,cond):
    lat=torch.cat([x,x],0).to(next(teacher_tf.parameters()).dtype)
    o=teacher_tf_call(lat,cond["ehs"],ts,cond["ex"]).float()
    vu,vc=o.chunk(2,0)
    return vu+GUID*(vc-vu)
def normed_mse(a,b,eps=1e-8):
    num=(a.float()-b.float()).square().flatten(1).mean(dim=1)
    den=b.float().square().flatten(1).mean(dim=1).clamp_min(eps)
    return (num/den).mean()
def prompt_tangent_loss():
    if teacher_tf is None or not ptd_pairs:
        z=_zero_loss(); return z,z,z,"none"
    row=random.choice(ptd_pairs)
    x,ts=ptd_state()
    s_pos=guided_student_condition(x,ts,row["pos"])
    s_neg=guided_student_condition(x,ts,row["neg"])
    delta_s=s_pos-s_neg
    with torch.no_grad():
        t_pos=guided_teacher_condition(x,ts,row["pos"])
        t_neg=guided_teacher_condition(x,ts,row["neg"])
        delta_t=t_pos-t_neg
    loss=normed_mse(delta_s,delta_t)
    return loss,tensor_rms(delta_t).detach(),tensor_rms(delta_s.detach()).detach(),row.get("id","pair")
def interaction_tangent_loss():
    if teacher_tf is None or not ptd_interactions:
        z=_zero_loss(); return z,z,z,"none"
    row=random.choice(ptd_interactions)
    x,ts=ptd_state()
    s_ab=guided_student_condition(x,ts,row["ab_c"])
    s_a=guided_student_condition(x,ts,row["a_c"])
    s_b=guided_student_condition(x,ts,row["b_c"])
    s_0=guided_student_condition(x,ts,row["null_c"])
    inter_s=s_ab-s_a-s_b+s_0
    with torch.no_grad():
        t_ab=guided_teacher_condition(x,ts,row["ab_c"])
        t_a=guided_teacher_condition(x,ts,row["a_c"])
        t_b=guided_teacher_condition(x,ts,row["b_c"])
        t_0=guided_teacher_condition(x,ts,row["null_c"])
        inter_t=t_ab-t_a-t_b+t_0
    loss=normed_mse(inter_s,inter_t)
    return loss,tensor_rms(inter_t).detach(),tensor_rms(inter_s.detach()).detach(),row.get("id","interaction")
def grad_rms(loss, params):
    try:
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    except RuntimeError:
        return 0.0
    total=0.0; count=0
    for g in grads:
        if g is None:
            continue
        gf=g.detach().float()
        total+=gf.square().sum().item()
        count+=gf.numel()
    return (total/max(count,1))**0.5 if count else 0.0
def aux_weight(base_loss, aux_loss, fallback, target_ratio, max_weight):
    if target_ratio<=0:
        return fallback,0.0,0.0
    br=grad_rms(base_loss,trainable)
    ar=grad_rms(aux_loss,trainable)
    if br<=0 or ar<=0:
        return fallback,br,ar
    return min(max_weight,target_ratio*br/ar),br,ar
@torch.no_grad()
def ptd_sanity_report():
    if teacher_tf is None:
        log("[ptd-sanity] skipped: no teacher")
        return
    def summarize(rows, kind):
        for row in rows:
            tsep=[]; ssep=[]; losses=[]
            for _ in range(max(1,PTD_SANITY_SAMPLES)):
                x,ts=ptd_state()
                if kind=="pair":
                    s_pos=guided_student_condition(x,ts,row["pos"])
                    s_neg=guided_student_condition(x,ts,row["neg"])
                    t_pos=guided_teacher_condition(x,ts,row["pos"])
                    t_neg=guided_teacher_condition(x,ts,row["neg"])
                    ds=s_pos-s_neg; dt=t_pos-t_neg
                else:
                    s_ab=guided_student_condition(x,ts,row["ab_c"])
                    s_a=guided_student_condition(x,ts,row["a_c"])
                    s_b=guided_student_condition(x,ts,row["b_c"])
                    s_0=guided_student_condition(x,ts,row["null_c"])
                    t_ab=guided_teacher_condition(x,ts,row["ab_c"])
                    t_a=guided_teacher_condition(x,ts,row["a_c"])
                    t_b=guided_teacher_condition(x,ts,row["b_c"])
                    t_0=guided_teacher_condition(x,ts,row["null_c"])
                    ds=s_ab-s_a-s_b+s_0; dt=t_ab-t_a-t_b+t_0
                tsep.append(float(tensor_rms(dt).item()))
                ssep.append(float(tensor_rms(ds).item()))
                losses.append(float(normed_mse(ds,dt).item()))
            log(f"[ptd-sanity] {kind} {row.get('id',kind)} teacher_sep={sum(tsep)/len(tsep):.4e} student_sep={sum(ssep)/len(ssep):.4e} normed={sum(losses)/len(losses):.4e}")
    summarize(ptd_pairs,"pair")
    summarize(ptd_interactions,"interaction")
def roll8(ti,k,grad_last):
    ehs,ex=ehs_pool[ti]; x=traj[(ti,k)]["hs"][:1].float()
    for j in range(1,M8+1):
        r=k+j-1; d=traj[(ti,r)]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
        ug=j>(M8-grad_last); ctx=torch.enable_grad() if ug else torch.no_grad()
        with ctx:
            o=tf_call(lat,ehs,d["ts"],ex).float(); vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu); x=x+(SIG[r+1]-SIG[r])*v
        if not ug: x=x.detach()
    return x
@torch.no_grad()
def vae_decode_ng(x):
    return vae.decode((x/VSCALE).to(vae.dtype),return_dict=False)[0].float().clamp(-1,1)
def vae_decode(x):
    return vae.decode((x/VSCALE).to(vae.dtype),return_dict=False)[0].float().clamp(-1,1)
SOB=torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],[[-1,-2,-1],[0,0,0],[1,2,1]]],dtype=torch.float32,device=dev).unsqueeze(1)
def lum(img): return 0.299*img[:,0:1]+0.587*img[:,1:2]+0.114*img[:,2:3]
def edge_map(img): return F.conv2d(lum(img),SOB.to(img.dtype),padding=1)
def lowpass(img):
    k=max(3,LOWPASS_K|1)
    return F.avg_pool2d(img,k,stride=1,padding=k//2)
def highpass(img): return img-lowpass(img)
def image_weight_at(global_step):
    if IMAGE_LOSS_EVERY<=0 or global_step<IMAGE_START:
        return 0.0
    if IMAGE_RAMP<=0:
        return 1.0
    return min(1.0,max(0.0,(global_step-IMAGE_START)/float(IMAGE_RAMP)))
def roll_to_x0(ti,k,grad_last):
    ehs,ex=ehs_pool[ti]; x=traj[(ti,k)]["hs"][:1].float(); n=K-k
    for j in range(1,n+1):
        r=k+j-1; d=traj[(ti,r)]; lat=torch.cat([x,x],0).to(d["hs"].dtype)
        ug=j>(n-grad_last); ctx=torch.enable_grad() if ug else torch.no_grad()
        with ctx:
            o=tf_call(lat,ehs,d["ts"],ex).float(); vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu); x=x+(SIG[r+1]-SIG[r])*v
        if not ug: x=x.detach()
    return x
def patch_loss(a,b):
    if LP is None or PATCH_COUNT<=0:
        return a.new_zeros(())
    k=min(PATCH_SIZE,a.shape[-1],a.shape[-2])
    with torch.no_grad():
        e=edge_map(b).abs().sum(1,keepdim=True); stride=max(1,k//2)
        pooled=F.avg_pool2d(e,k,stride=stride)
        idx=pooled.flatten().topk(min(PATCH_COUNT,pooled.numel())).indices; nw=pooled.shape[-1]
        coords=[((int(i)//nw)*stride,(int(i)%nw)*stride) for i in idx.tolist()]
    loss=a.new_zeros(())
    for y,x in coords:
        y=min(y,a.shape[-2]-k); x=min(x,a.shape[-1]-k)
        loss=loss+LP(a[:,:,y:y+k,x:x+k],b[:,:,y:y+k,x:x+k]).mean()
    return loss/max(1,len(coords))
image_states=[(ti,k) for ti in calib_tis for k in range(max(0,IMAGE_MIN_K),K)]
if IMAGE_MAX_STATES>0 and len(image_states)>IMAGE_MAX_STATES:
    image_states=sorted(random.Random(23).sample(image_states,IMAGE_MAX_STATES))
bell_value_states=[(ti,k) for ti in calib_tis for k in range(K-1)]
def bellman_weight_at(global_step):
    if not BELL_ENABLE or BELL_ACTOR_LAMBDA<=0 or global_step<BELL_START:
        return 0.0
    if BELL_RAMP<=0:
        return BELL_ACTOR_LAMBDA
    t=min(1.0,max(0.0,(global_step-BELL_START)/float(BELL_RAMP)))
    return BELL_ACTOR_LAMBDA*t
def state_features(x,k):
    x=x.float()
    if x.ndim>=3:
        reduce_dims=tuple(range(1,x.ndim-1))
        mean=x.mean(dim=reduce_dims)
        std=x.std(dim=reduce_dims,unbiased=False)
    elif x.ndim==2:
        mean=x
        std=torch.zeros_like(x)
    else:
        mean=x.reshape(1,-1)
        std=torch.zeros_like(mean)
    b=mean.shape[0]
    kk=float(k)/max(K-1,1)
    sig=SIG[min(max(int(k),0),K)].float()
    scalars=torch.tensor([kk,float(sig.item()),float(torch.log(sig.clamp_min(1e-8)).item())],
                         device=mean.device,dtype=mean.dtype).view(1,3).expand(b,-1)
    return torch.cat([mean,std,scalars],dim=-1)
class LatentValueCritic(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net=nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,hidden),
            nn.SiLU(),
            nn.Linear(hidden,max(64,hidden//2)),
            nn.SiLU(),
            nn.Linear(max(64,hidden//2),1),
        )
    def forward(self, feat):
        return self.net(feat.float())
def bellman_one_step(ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; x=d["hs"][:1].float()
    lat=torch.cat([x,x],0).to(d["hs"].dtype)
    o=tf_call(lat,ehs,d["ts"],ex).float()
    vu,vc=o.chunk(2,0); v=vu+GUID*(vc-vu)
    return x+(SIG[k+1]-SIG[k])*v
def guided_next_from_velocity(ti,k,v):
    vu,vc=v.float().chunk(2,0)
    vg=vu+GUID*(vc-vu)
    return traj[(ti,k)]["hs"][:1].float()+(SIG[k+1]-SIG[k])*vg
def bellman_relative_value(x,xT):
    rel=(x.float()-xT.float()).pow(2).mean()/xT.float().pow(2).mean().clamp_min(1e-8)
    return -torch.sqrt(rel.clamp(0.0,BELL_TARGET_CLAMP))
bell_critic=None; bell_opt=None; bell_last_loss=0.0; bell_last_target=0.0
if BELL_ENABLE:
    sample_ti=calib_tis[0] if calib_tis else val_tis[0]
    feat_dim=state_features(traj[(sample_ti,0)]["hs"][:1],0).shape[-1]
    bell_critic=LatentValueCritic(feat_dim,BELL_HIDDEN).to(dev)
    bell_opt=torch.optim.AdamW(bell_critic.parameters(),lr=BELL_CRITIC_LR,betas=(0.9,0.95),weight_decay=0.0)
    if RESUME_STATE is not None and RESUME_STATE.get("bellman_critic"):
        bell_critic.load_state_dict(RESUME_STATE["bellman_critic"])
        if RESUME_STATE.get("bellman_optimizer"):
            bell_opt.load_state_dict(RESUME_STATE["bellman_optimizer"])
            optimizer_state_to_device(bell_opt)
        log(f"[bellman] loaded critic from {RESUME}")
    log(f"[bellman] critic enabled feat_dim={feat_dim} hidden={BELL_HIDDEN} actor_lambda={BELL_ACTOR_LAMBDA:g} gamma={BELL_GAMMA:g}")
def train_bellman_critic(global_step):
    global bell_last_loss, bell_last_target
    if bell_critic is None or not calib_chunks or BELL_CRITIC_EVERY<=0 or global_step%BELL_CRITIC_EVERY!=0:
        return bell_last_loss,bell_last_target
    bell_critic.train()
    loss_out=0.0; target_out=0.0
    for _ in range(max(1,BELL_CRITIC_UPDATES)):
        ti,k=random.choice(calib_chunks)
        with torch.no_grad():
            xS=roll8(ti,k,0).detach()
            xT=traj[(ti,k+M8)]["hs"][:1].float().detach()
            target=bellman_relative_value(xS,xT).detach().view(1,1)
            teacher_target=xT.new_zeros((1,1))
            if BELL_NOISE_WEIGHT>0 and BELL_NOISE_STD>0:
                xn=xT+torch.randn_like(xT)*BELL_NOISE_STD*xT.float().std().clamp_min(1e-6)
                noise_target=bellman_relative_value(xn,xT).detach().view(1,1)
        pred=bell_critic(state_features(xS,k+M8))
        loss=F.mse_loss(pred,target)
        if BELL_TEACHER_WEIGHT>0:
            loss=loss+BELL_TEACHER_WEIGHT*F.mse_loss(bell_critic(state_features(xT,k+M8)),teacher_target)
        if BELL_NOISE_WEIGHT>0 and BELL_NOISE_STD>0:
            loss=loss+BELL_NOISE_WEIGHT*F.mse_loss(bell_critic(state_features(xn,k+M8)),noise_target)
        bell_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bell_critic.parameters(),1.0)
        bell_opt.step()
        loss_out=float(loss.detach().item()); target_out=float(target.detach().item())
    bell_last_loss, bell_last_target=loss_out,target_out
    return loss_out,target_out
def bellman_actor_loss(global_step):
    if bell_critic is None or not bell_value_states or BELL_ACTOR_EVERY<=0 or global_step%BELL_ACTOR_EVERY!=0:
        z=trainable[0].new_zeros(()) if trainable else torch.zeros((),device=dev)
        return z,z
    ti,k=random.choice(bell_value_states)
    x1=bellman_one_step(ti,k)
    req=[p.requires_grad for p in bell_critic.parameters()]
    for p in bell_critic.parameters(): p.requires_grad_(False)
    bell_critic.eval()
    v=bell_critic(state_features(x1,k+1)).mean()
    for p,rq in zip(bell_critic.parameters(),req): p.requires_grad_(rq)
    return -v, v.detach()
def bellman_actor_loss_from_next(x1,k_next):
    if bell_critic is None:
        z=trainable[0].new_zeros(()) if trainable else torch.zeros((),device=dev)
        return z,z
    req=[p.requires_grad for p in bell_critic.parameters()]
    for p in bell_critic.parameters(): p.requires_grad_(False)
    bell_critic.eval()
    v=bell_critic(state_features(x1,k_next)).mean()
    for p,rq in zip(bell_critic.parameters(),req): p.requires_grad_(rq)
    return -v, v.detach()
def snapshot_quant_state():
    return {"mods":[(m.weight_master.detach().cpu().clone(),m.log_scales.detach().cpu().clone()) for m in ste_mods],
            "islands":{n:p.detach().cpu().clone() for n,p,_ in island_params}}
def restore_quant_state(snap):
    for m,(wm,ls) in zip(ste_mods,snap["mods"]):
        m.weight_master.data.copy_(wm.to(m.weight_master.device,dtype=m.weight_master.dtype))
        m.log_scales.data.copy_(ls.to(m.log_scales.device,dtype=m.log_scales.dtype))
    for n,p,_ in island_params:
        src=snap["islands"].get(n)
        if src is not None: p.data.copy_(src.to(p.device,dtype=p.dtype))
def load_comp_anchor_state(path):
    ck=torch_load_cpu(path)
    if ck.get("format")==FULL_STATE_FORMAT and isinstance(ck.get("quant"),dict):
        ck=ck["quant"]
    return load_compact_model(ck,path)
comp_lp={}
if COMP_CKPT and LAM_COMP>0 and image_states:
    comp_states=image_states if COMP_ANCHOR_MAX<=0 else image_states[:COMP_ANCHOR_MAX]
    snap=snapshot_quant_state(); load_comp_anchor_state(COMP_CKPT); BM.set_bonsai_forward_blend_alpha(tf,1.0)
    with torch.no_grad():
        for ti,k in comp_states:
            comp_lp[(ti,k)]=lowpass(vae_decode_ng(roll_to_x0(ti,k,0))).detach().cpu()
    restore_quant_state(snap); del snap; torch.cuda.empty_cache()
    log(f"[image-anchor] ckpt={COMP_CKPT} states={len(comp_lp)} lpips={LP is not None}")
@torch.no_grad()
def eval_c8():
    bk={'E':[0.,0.],'M':[0.,0.],'L':[0.,0.]}
    for (ti,k) in EVALC:
        xS=roll8(ti,k,0); xT=traj[(ti,k+M8)]["hs"][:1].float(); e=(xS-xT).pow(2).sum().item(); t=xT.pow(2).sum().item()
        b=bucket(k); bk[b][0]+=e; bk[b][1]+=t
    return {b:(v[0]/v[1])**0.5 if v[1]>0 else 0 for b,v in bk.items()}
def fmt(d): return f"E {d['E']:.3f} M {d['M']:.3f} L {d['L']:.3f}"
def vel_loss(ti,k):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; v=tf_call(d["hs"],ehs,d["ts"],ex); return F.mse_loss(v.float(),d["vT"].float())
prompt_ti_groups={}
for ti in calib_tis:
    prompt_ti_groups.setdefault(ti_meta[ti]["prompt"],[]).append(ti)
stein_groups=[v for v in prompt_ti_groups.values() if len(v)>=2]
stein_projection_cache={}
def _zero_loss():
    return trainable[0].new_zeros(()) if trainable else torch.zeros((),device=dev)
def student_state_path_loss():
    if teacher_tf is None or not calib_tis:
        z=_zero_loss(); return z,z
    horizon=max(1,min(JEFFREYS_HORIZON,K-1))
    max_k=max(0,K-horizon-1)
    min_k=min(max(0,JEFFREYS_MIN_K),max_k)
    ti=random.choice(calib_tis)
    k=random.randint(min_k,max_k) if max_k>=min_k else 0
    x=traj[(ti,k)]["hs"][:1].float().detach()
    loss=x.new_zeros(())
    resid=[]
    for j in range(horizon):
        r=k+j
        vS=guided_student_velocity(x,ti,r)
        vT=guided_teacher_velocity(x.detach(),ti,r)
        loss=loss+F.mse_loss(vS.float(),vT.float())
        resid.append((vS.detach().float()-vT.float()).square().mean())
        x=x+(SIG[r+1]-SIG[r])*vS
    return loss/float(horizon), torch.stack(resid).mean().sqrt().detach()
def stein_ramp_at(branch_step):
    if not STEIN_ENABLE or branch_step<STEIN_START:
        return 0.0
    if STEIN_RAMP<=0:
        return 1.0
    return min(1.0,max(0.0,(branch_step-STEIN_START)/float(STEIN_RAMP)))
def cached_stein_projection(states):
    flat_dim=int(states[0].numel())
    key=(flat_dim,STEIN_PROJ_DIM,STEIN_PROJ_SEED)
    proj=stein_projection_cache.get(key)
    if proj is None or proj.device != states.device:
        proj=make_rademacher_projection(flat_dim,STEIN_PROJ_DIM,device=states.device,seed=STEIN_PROJ_SEED)
        stein_projection_cache[key]=proj
    return proj
def stein_particle_loss(path_rms, branch_step):
    if teacher_tf is None or not stein_groups:
        z=_zero_loss(); return z,z,0.0
    horizon=max(1,min(STEIN_HORIZON,K-2))
    max_k=max(0,K-horizon-1)
    min_k=min(max(0,STEIN_MIN_K),max_k)
    group=random.choice(stein_groups)
    if len(group)>=STEIN_PARTICLES:
        tis=random.sample(group,STEIN_PARTICLES)
    else:
        tis=[random.choice(group) for _ in range(max(2,STEIN_PARTICLES))]
    start_k=random.randint(min_k,max_k) if max_k>=min_k else 0
    score_k=start_k+horizon
    states=[]
    for ti in tis:
        x=traj[(ti,start_k)]["hs"][:1].float().detach()
        for j in range(horizon):
            r=start_k+j
            vS=guided_student_velocity(x,ti,r)
            x=x+(SIG[r+1]-SIG[r])*vS
        states.append(x)
    states=torch.cat(states,0)
    scores=[]
    with torch.no_grad():
        sigma=SIG[score_k].float().clamp_min(1e-6)
        for i,ti in enumerate(tis):
            vT=guided_teacher_velocity(states[i:i+1].detach(),ti,score_k)
            scores.append(flow_velocity_to_score(states[i:i+1].detach(),vT,sigma))
    scores=torch.cat(scores,0)
    proj=cached_stein_projection(states)
    field=stein_imq_field(
        states,
        scores,
        proj,
        kernel_offset=STEIN_KERNEL_OFFSET,
        beta=STEIN_BETA,
        normalize_field=bool(STEIN_NORMALIZE_FIELD),
    )
    field_rms=tensor_rms(field).detach()
    loss=stein_transport_loss(states,field)
    ramp=stein_ramp_at(branch_step)
    if torch.is_tensor(path_rms) and float(path_rms.item())>0 and float(field_rms.item())>0:
        weight=STEIN_TARGET_RATIO*ramp*float(path_rms.item())/float(field_rms.item())
    else:
        weight=STEIN_TARGET_RATIO*ramp
    weight=min(STEIN_MAX_WEIGHT,max(0.0,weight))
    return loss,field_rms,weight
anchor_v={}
if ANCHOR_LAMBDA>0:
    anchor_states=[s for s in calib_states if ti_meta.get(s[0],{}).get("anchor")]
    if ANCHOR_MAX_STATES>0 and len(anchor_states)>ANCHOR_MAX_STATES:
        anchor_states=random.Random(0).sample(anchor_states,ANCHOR_MAX_STATES)
        anchor_states=sorted(anchor_states)
    with torch.no_grad():
        asv=[m.forward_blend_alpha for m in ste_mods]; BM.set_bonsai_forward_blend_alpha(tf,1.0)
        for ti,k in anchor_states:
            d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]
            anchor_v[(ti,k)]=tf_call(d["hs"],ehs,d["ts"],ex).detach().float().cpu()
        for m,a in zip(ste_mods,asv): m.forward_blend_alpha=a
    log(f"[anchor] lambda={ANCHOR_LAMBDA:g} categories={sorted(ANCHOR_CATEGORIES)} states={len(anchor_v)}")
if PTD_SANITY_ONLY:
    ptd_sanity_report()
    if DDP:
        dist.barrier(); dist.destroy_process_group()
    sys.exit(0)
def vel_anchor_loss(ti,k,need_next=False):
    d=traj[(ti,k)]; ehs,ex=ehs_pool[ti]; v=tf_call(d["hs"],ehs,d["ts"],ex)
    lv=F.mse_loss(v.float(),d["vT"].float())
    xnext=guided_next_from_velocity(ti,k,v) if need_next and k<K-1 else None
    av=anchor_v.get((ti,k))
    if av is None:
        return lv, v.new_zeros(()), False, xnext
    return lv, F.mse_loss(v.float(),av.to(v.device,dtype=torch.float32)), True, xnext
def sampled_codebook_modules():
    mods=[m for m in ste_mods if m.weight_master.requires_grad]
    n=CODEBOOK_SAMPLE_MODULES
    if n<=0 or n>=len(mods):
        return mods
    return random.sample(mods,n)
def codebook_geometry_loss(mods, need_code, need_margin):
    if not need_code and not need_margin:
        z=trainable[0].new_zeros(()) if trainable else torch.zeros((),device=dev)
        return z,z,0
    lc=None; lm=None; denom=0
    eps=1e-8
    for m in mods:
        wg,_,in_features=BM._group_view_2d(m.weight_master.float(),m.group_size)
        out,groups,g=wg.shape
        mask=BM._valid_group_mask(out,groups,g,in_features,wg.device)
        scales=m._scales().detach().float().to(wg.device).unsqueeze(-1)
        n=mask.float().sum().clamp_min(1.0)
        if need_code:
            codes=m._codes_from_matrix_detached(m.weight_master).to(wg.device).float()
            qg=codes*scales
            v=((wg-qg).pow(2)/(scales.pow(2)+eps)).masked_fill(~mask,0.0).sum()
            lc=v if lc is None else lc+v
        if need_margin and getattr(m,"mode","")=="ternary":
            tau=float(getattr(m,"threshold_factor",0.5))
            dist=(wg.abs()-tau*scales).abs()/(scales+eps)
            v=torch.exp(-dist/max(MARGIN_TEMP,1e-6)).masked_fill(~mask,0.0).sum()
            lm=v if lm is None else lm+v
        denom+=float(n.item())
    z=trainable[0].new_zeros(()) if trainable else torch.zeros((),device=dev)
    if lc is None: lc=z
    else: lc=lc/max(denom,1.0)
    if lm is None: lm=z
    else: lm=lm/max(denom,1.0)
    return lc,lm,len(mods)
@torch.no_grad()
def forensics():   # the Prism-pattern check: support moves, sign flips low, proj_out frozen, caption_norm compensates
    zf=0.;n=0;sf=0.;sa=0.; sc=[]
    for i,m in enumerate(ste_mods):
        code=m._codes_from_matrix_detached(m.weight_master).float(); nz=code.numel()
        zf+=(code==0).float().sum().item(); n+=nz
        cs=torch.sign(code).cpu(); init_act=init_codesign[i]!=0; both=(code!=0).cpu()&init_act   # flip = active at BOTH, sign differs (not an activation)
        sf+=((cs!=init_codesign[i])&both).float().sum().item(); sa+=both.float().sum().item()
        sc.append(m._scales().mean().item())
    isl={}
    for n2,p,_ in island_params:
        key="caption_norm" if "caption_norm" in n2 else ("proj_out" if "proj_out" in n2 else ("scale_shift" if "scale_shift" in n2 else "other"))
        isl.setdefault(key,0.); isl[key]+=(p.detach()-init_island_val[n2]).norm().item()
    smean=sum(sc)/len(sc)
    return zf/n, (sf/max(sa,1)), smean, isl
@torch.no_grad()
def render(tag):
    asv=[m.forward_blend_alpha for m in ste_mods]; BM.set_bonsai_forward_blend_alpha(tf,1.0)
    qs=[pipe(prompt=p,num_inference_steps=K,height=512,width=512,guidance_scale=GUID,generator=torch.Generator(dev).manual_seed(9000+j)).images[0] for j,p in enumerate(hardP)]
    for m,a in zip(ste_mods,asv): m.forward_blend_alpha=a
    W=512; M=Image.new("RGB",(W*len(hardP),W*2),(20,20,20))
    for j in range(len(hardP)): M.paste(refs[j],(j*W,0)); M.paste(qs[j],(j*W,W))
    M.save(f"render_{TAG}_{tag}.png"); log(f"[render] render_{TAG}_{tag}.png")
    torch.cuda.empty_cache()
@torch.no_grad()
def save_ckpt(global_step, force_full=False):
    full_due=bool(SAVE_FULL_STATE and (force_full or (FULL_CKPT_EVERY>0 and global_step%FULL_CKPT_EVERY==0) or shutdown["signum"] is not None or global_step==BASE_STEP+STEPS))
    rank_rng=collect_rank_rng_states() if full_due else None
    if not MAIN: return
    compact=compact_payload(global_step)
    compact_latest=f"ckpt_{TAG}.pt"
    atomic_torch_save(compact,compact_latest)
    compact_step=None
    if SAVE_STEPS:
        compact_step=f"ckpt_{TAG}_s{global_step}.pt"
        atomic_torch_save(compact,compact_step)
    full_latest=None; full_step=None
    if full_due:
        full_latest=f"state_{TAG}.pt"
        full={"format":FULL_STATE_FORMAT,"version":1,"tag":TAG,"global_step":global_step,
              "saved_at_unix":time.time(),"base_step":BASE_STEP,"best":best[0],
              "config":run_config(),"quant":compact,"quant_names":[n for n,_ in ste_named_mods],
              "island_names":[n for n,_,_ in island_params],
              "transformer":to_cpu(tf.state_dict()),
              "optimizer":to_cpu(opt.state_dict()),
              "bellman_critic":to_cpu(bell_critic.state_dict()) if bell_critic is not None else None,
              "bellman_optimizer":to_cpu(bell_opt.state_dict()) if bell_opt is not None else None,
              "rank_rng":rank_rng,
              "shutdown_signal":shutdown["signum"]}
        atomic_torch_save(full,full_latest)
        if SAVE_STEPS:
            full_step=f"state_{TAG}_s{global_step}.pt"
            atomic_torch_save(full,full_step)
    manifest={"tag":TAG,"global_step":global_step,"saved_at_unix":time.time(),
              "compact_latest":compact_latest,"compact_step":compact_step,
              "full_latest":full_latest,"full_step":full_step,
              "save_full_state":bool(SAVE_FULL_STATE),"best":best[0],
              "shutdown_signal":shutdown["signum"],"config":run_config()}
    atomic_json_save(manifest,f"manifest_{TAG}.json")
    log(f"[ckpt] step {global_step} compact={compact_latest}{' full='+full_latest if full_latest else ''}")

best=[float(RESUME_STATE.get("best",9e9)) if RESUME_STATE is not None else 9e9]; t0=time.time()
if MAIN:
    BM.set_bonsai_forward_blend_alpha(tf,1.0); zf,sf,sm,isl=forensics(); init_c8=eval_c8()
    if START_HARD: best[0]=min(best[0],max(init_c8.values()))
    log(f"[init a=1] resume_step {BASE_STEP} chunk8 {fmt(init_c8)} | zero {zf:.3f} flip {sf:.4f} scale {sm:.3f}"); render(f"s{BASE_STEP}" if BASE_STEP else "s0")
if SAVE_INITIAL_STATE and (START_HARD or BASE_STEP):
    save_ckpt(BASE_STEP,force_full=True)
stopped=False
for step in range(1,STEPS+1):
    step_t=time.time()
    global_step=BASE_STEP+step
    a=1.0 if START_HARD else min(1.0, global_step/float(RAMP)); BM.set_bonsai_forward_blend_alpha(tf,a)
    rho=0.0 if (a<1.0 or global_step<P2START) else (0.005 if global_step<3000 else (0.02 if global_step<8000 else 0.05))
    if PROX_RHO_OVERRIDE:
        rho=float(PROX_RHO_OVERRIDE)
    thr=threshold_boost_at(global_step) if OPT_KIND=="bonsai_qat8" else 0.0
    codew=codebook_lambda_at(step)
    marginw=margin_lambda_at(step)
    if OPT_KIND=="bonsai_qat8":
        opt.set_prox_rho(rho); opt.set_threshold_boost(thr)
    bellcrit,belltarg=train_bellman_critic(global_step)
    opt.zero_grad(set_to_none=True)
    r=random.random(); cb='E' if r<0.3 else ('M' if r<0.6 else 'L')
    hb=bool(HEARTBEAT_EVERY>0 and (global_step%HEARTBEAT_EVERY==0 or step==1))
    if hb and MAIN:
        log(f"[hb s{global_step}] begin bucket={cb} a={a:.2f} rho={rho} thr={thr:.3f}")
    imgw=image_weight_at(global_step)
    bellw=bellman_weight_at(global_step)
    do_image=bool(imgw>0 and IMAGE_LOSS_EVERY>0 and global_step%IMAGE_LOSS_EVERY==0 and image_states)
    z=trainable[0].new_zeros(()) if trainable else torch.zeros((),device=dev)
    bell_next=None; bell_next_k=0
    if do_image:
        Lv=La=z; L=z; logm=0.0; has_anchor=False
    else:
        ti,k=random.choice(cs_b[cb] or calib_states)
        need_bell_next=bool(bellw>0 and BELL_ACTOR_EVERY>0 and global_step%BELL_ACTOR_EVERY==0 and k<K-1)
        Lv,La,has_anchor,bell_next=vel_anchor_loss(ti,k,need_next=need_bell_next)
        bell_next_k=k+1
        L=0.7*Lv+ANCHOR_LAMBDA*La; logm=0.0
        if global_step>=P2START:
            cs=cc_b[cb] or calib_chunks; cti,ck=random.choice(cs); xS=roll8(cti,ck,KBPTD); xT=traj[(cti,ck+M8)]["hs"][:1].float(); Lm=F.mse_loss(xS,xT); L=0.25*Lv+1.0*Lm+ANCHOR_LAMBDA*La; logm=Lm.item()
    z=L.new_zeros(())
    Lcomp=Ldetail=Ledge=Lpatch=Lpix=Lbell=Vbell=Ljeff=Lstein=Lptd=Lint=z
    jeff_rms=stein_field=z; steinw=0.0
    ptd_tsep=ptd_ssep=int_tsep=int_ssep=z
    ptdw=intw=0.0; ptd_br=ptd_ar=int_br=int_ar=0.0; ptd_id=int_id="none"
    imgw=image_weight_at(global_step)
    if do_image:
        torch.cuda.empty_cache()
        iti,ik=random.choice(list(comp_lp.keys()) if comp_lp else image_states)
        xi=roll_to_x0(iti,ik,IMAGE_BPTT); IS=vae_decode(xi)
        with torch.no_grad():
            IT=vae_decode_ng(x0T[iti].float()); CLP=comp_lp.get((iti,ik))
            CLP=lowpass(IT).detach() if CLP is None else CLP.to(IS.device,dtype=IS.dtype)
        if LAM_COMP>0: Lcomp=F.l1_loss(lowpass(IS),CLP)
        if LAM_DETAIL>0: Ldetail=F.l1_loss(highpass(IS),highpass(IT))
        if LAM_EDGE>0: Ledge=(edge_map(IS)-edge_map(IT)).abs().mean()
        if LAM_PATCH>0: Lpatch=patch_loss(IS,IT)
        if LAM_PIX>0: Lpix=(IS-IT).abs().mean()
        L=L+imgw*(LAM_COMP*Lcomp+LAM_DETAIL*Ldetail+LAM_EDGE*Ledge+LAM_PATCH*Lpatch+LAM_PIX*Lpix)
    if bellw>0 and not do_image and bell_next is not None:
        Lbell,Vbell=bellman_actor_loss_from_next(bell_next,bell_next_k)
        L=L+bellw*Lbell
    if JEFFREYS_ENABLE and teacher_tf is not None and not do_image and JEFFREYS_EVERY>0 and global_step%JEFFREYS_EVERY==0:
        Ljeff,jeff_rms=student_state_path_loss()
        L=L+JEFFREYS_LAMBDA*Ljeff
    if STEIN_ENABLE and teacher_tf is not None and not do_image and STEIN_EVERY>0 and global_step%STEIN_EVERY==0:
        Lstein,stein_field,steinw=stein_particle_loss(jeff_rms,step)
        L=L+steinw*Lstein
    if PTD_ENABLE and teacher_tf is not None and not do_image and PTD_EVERY>0 and global_step%PTD_EVERY==0:
        Lptd,ptd_tsep,ptd_ssep,ptd_id=prompt_tangent_loss()
        ptdw,ptd_br,ptd_ar=aux_weight(L,Lptd,PTD_LAMBDA,PTD_TARGET_RATIO,PTD_MAX_WEIGHT)
        L=L+ptdw*Lptd
    if PTD_ENABLE and teacher_tf is not None and not do_image and PTD_INT_EVERY>0 and global_step%PTD_INT_EVERY==0:
        Lint,int_tsep,int_ssep,int_id=interaction_tangent_loss()
        intw,int_br,int_ar=aux_weight(L,Lint,PTD_INT_LAMBDA,PTD_INT_TARGET_RATIO,PTD_INT_MAX_WEIGHT)
        L=L+intw*Lint
    Lcode,Lmargin,ngeom=codebook_geometry_loss(sampled_codebook_modules(),codew>0,marginw>0)
    L=L+codew*Lcode+marginw*Lmargin
    L.backward()
    if DDP:
        for p in trainable:
            if p.grad is None: p.grad=torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    freeze_attention_grads(step)
    gn=torch.nn.utils.clip_grad_norm_(trainable,0.5)
    if torch.isfinite(gn): opt.step()
    if OPT_KIND!="bonsai_qat8": prox_pull(rho)
    if hb and MAIN:
        log(f"[hb s{global_step}] done dt={time.time()-step_t:.1f}s Lv={Lv.item():.2e} Lm={logm:.2e} La={La.item():.2e}{'*' if has_anchor else ''} Lcomp={Lcomp.item():.2e}@{imgw:.2f} Ldet={Ldetail.item():.2e} Ledge={Ledge.item():.2e} Lpatch={Lpatch.item():.2e} Lbell={Lbell.item():.2e}@{bellw:.3g} Vbell={Vbell.item():.2e} Cbell={bellcrit:.2e}/{belltarg:.2e} Ljeff={Ljeff.item():.2e}@{JEFFREYS_LAMBDA:.3g} Jrms={jeff_rms.item():.2e} Lstein={Lstein.item():.2e}@{steinw:.3g} Srms={stein_field.item():.2e} Lptd={Lptd.item():.2e}@{ptdw:.3g}/{ptd_id} Psep={ptd_tsep.item():.2e}/{ptd_ssep.item():.2e} Pint={Lint.item():.2e}@{intw:.3g}/{int_id} Isep={int_tsep.item():.2e}/{int_ssep.item():.2e} Lcode={Lcode.item():.2e} Lmargin={Lmargin.item():.2e} ngeom={ngeom} gn={float(gn):.2f}")
    if global_step%REFRESH==0: refresh_scales()
    if (global_step%FORENSE==0 or step==1) and MAIN:
        zf,sf,sm,isl=forensics(); c8=eval_c8(); mx=max(c8.values()); best[0]=min(best[0],mx) if a>=1.0 else best[0]
        isls=" ".join(f"{kk}{vv:.2f}" for kk,vv in isl.items())
        log(f"[s{global_step}] a{a:.2f} rho{rho} thr{thr:.3f} Lv{Lv.item():.2e} Lm{logm:.2e} La{La.item():.2e}{'*' if has_anchor else ''} Lcomp{Lcomp.item():.2e}@{imgw:.2f} Ldet{Ldetail.item():.2e} Ledge{Ledge.item():.2e} Lpatch{Lpatch.item():.2e} Lbell{Lbell.item():.2e}@{bellw:.3g} Vbell{Vbell.item():.2e} Cbell{bellcrit:.2e}/{belltarg:.2e} Ljeff{Ljeff.item():.2e}@{JEFFREYS_LAMBDA:.3g} Jrms{jeff_rms.item():.2e} Lstein{Lstein.item():.2e}@{steinw:.3g} Srms{stein_field.item():.2e} Lptd{Lptd.item():.2e}@{ptdw:.3g}/{ptd_id} Psep{ptd_tsep.item():.2e}/{ptd_ssep.item():.2e} Pint{Lint.item():.2e}@{intw:.3g}/{int_id} Isep{int_tsep.item():.2e}/{int_ssep.item():.2e} Lcode{Lcode.item():.2e}@{codew:.3g} Lmargin{Lmargin.item():.2e}@{marginw:.3g} ngeom{ngeom} gn{gn:.1f} chunk8 {fmt(c8)} | zero {zf:.3f} flip {sf:.4f} scale {sm:.3f} [{isls}] best {best[0]:.3f} ({time.time()-t0:.0f}s)")
    if global_step%RENDER_EVERY==0 and a>=1.0 and MAIN: render(f"s{global_step}")
    if global_step%CKPT_EVERY==0 and a>=1.0: save_ckpt(global_step)
    if stop_requested():
        save_ckpt(global_step,force_full=True)
        if MAIN: log(f"[done] stopped after request {shutdown['signum']} at step {global_step}")
        stopped=True
        break
if not stopped and shutdown["signum"] is None:
    if MAIN: render("final")
    save_ckpt(BASE_STEP+STEPS)
    if MAIN: log("[done]")
if DDP: dist.barrier(); dist.destroy_process_group()
