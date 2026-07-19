"""Stage 10A — Sana weight forensics vs Prism signatures. Measure how our trained ternary q1024t
differs from FP16 base and from optimal PTQ, to see which Prism endpoint statistics we're missing.
Prism ternary targets: sign_flips ~0% (signs frozen from FP16), zero_frac ~38% (PTQ ~48%, densified
+10pts), scale_ratio ~0.625x PTQ, scale_std narrowed, proj_out delta 0% (frozen), norm_out delta
100-150% (main compensator), block modulation ~1-2%, late blocks move more than early.
ENV: CKPT=ckpt_ternary_preview_v0.pt
"""
import torch, torch.nn as nn, os, sys, json
os.environ["HF_HUB_OFFLINE"]="1"
from diffusers import SanaPipeline
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clark_quantize.sana_ckpt import island_values, load_compact_checkpoint, validate_quant_names
dev="cuda"; MODEL="models/Sana_1600M_512px"; G=112
CKPT=os.environ.get("CKPT","ckpt_ternary_preview_v0.pt")
def is_trunk_lin(n,m): return isinstance(m,nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) and any(s in n for s in ("to_q","to_k","to_v","to_out.0"))
def is_trunk_conv(n,m): return isinstance(m,nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n,m): return is_trunk_lin(n,m) or is_trunk_conv(n,m)
def is_island(n): return any(k in n for k in ("patch_embed","time_embed","caption_projection","caption_norm","scale_shift_table","norm_out","proj_out"))
def _blk(n):
    try: return int(n.split("transformer_blocks.")[1].split(".")[0])
    except: return -1

@torch.no_grad()
def ptq_ternary(W):  # closed-form optimal ternary at g=G -> codes[out,ng,G] (-1/0/1), scales[out,ng]
    o,i=W.shape; pad=(-i)%G
    Wp=torch.cat([W.float(),torch.zeros(o,pad,device=W.device,dtype=torch.float32)],1) if pad else W.float()
    Wg=Wp.view(o,Wp.shape[1]//G,G); a=Wg.abs()
    sorted_abs, sorted_idx=torch.sort(a,dim=-1,descending=True)
    prefix=torch.cumsum(sorted_abs,dim=-1)
    k=torch.arange(1,G+1,device=W.device,dtype=prefix.dtype).view(1,1,G)
    best_idx=torch.argmax(prefix.square()/k,dim=-1)
    scale=(torch.gather(prefix,-1,best_idx.unsqueeze(-1)).squeeze(-1)/(best_idx.to(prefix.dtype)+1)).clamp_min(1e-8)
    rank=torch.arange(G,device=W.device).view(1,1,G)
    active_sorted=rank<=best_idx.unsqueeze(-1)
    sorted_sign=torch.gather(torch.sign(Wg),-1,sorted_idx)
    sorted_codes=sorted_sign*active_sorted.to(Wg.dtype)
    codes=torch.zeros_like(Wg); codes.scatter_(-1,sorted_idx,sorted_codes)
    if pad: codes.reshape(o,-1)[:,i:]=0
    return codes, scale

ck=load_compact_checkpoint(CKPT)
pipe=SanaPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16); pipe.to(dev); tf=pipe.transformer
qmods=[(n,m) for n,m in tf.named_modules() if predicate(n,m)]
validate_quant_names(ck,[n for n,_ in qmods])
print(f"[forensics] {CKPT} step {ck['step']} | {len(qmods)} trunk mods",flush=True)

# global accumulators
tot=0; z_tr=0; z_ptq=0; flip=0; nz_tr=0; z2nz=0; nz2z=0
s_tr_all=[]; s_ptq_all=[]; blk_delta={}; blk_n={}
for (n,m),code_tr,ls in zip(qmods,ck["codes"],ck["log_scales"]):
    W=m.weight.data.reshape(m.weight.shape[0],-1).float().to(dev)         # FP16 base [out,in]
    o,i=W.shape; ng=i//G; Wg=W.view(o,ng,G)
    ct=code_tr.float().to(dev)                                            # trained codes [out,ng,G]
    st=torch.exp(ls.float().to(dev))                                     # trained scales [out,ng]
    cp,sp=ptq_ternary(W)                                                  # ptq codes/scales
    sign_base=torch.where(Wg>=0,1.,-1.)
    nz=(ct!=0)
    tot+=ct.numel(); z_tr+=(ct==0).sum().item(); z_ptq+=(cp==0).sum().item(); nz_tr+=nz.sum().item()
    flip+=((nz)&(torch.sign(ct)!=sign_base)).sum().item()
    z2nz+=((cp==0)&(ct!=0)).sum().item(); nz2z+=((cp!=0)&(ct==0)).sum().item()
    s_tr_all.append(st.flatten()); s_ptq_all.append(sp.flatten().to(dev))
    # block weight delta: ||W_trained_dequant - W_fp16|| / ||W_fp16||
    Wtr=(ct*st.unsqueeze(-1)).view(o,i); d=(Wtr-W).norm()/W.norm()
    b=_blk(n); blk_delta[b]=blk_delta.get(b,0.)+d.item(); blk_n[b]=blk_n.get(b,0)+1

st_all=torch.cat(s_tr_all); sp_all=torch.cat(s_ptq_all)
print("\n=== GLOBAL (trunk ternary) ===",flush=True)
print(f"  zero_frac trained   {z_tr/tot:.3f}   (PTQ {z_ptq/tot:.3f})   [Prism ~0.38 from ~0.48]",flush=True)
print(f"  0->nonzero          {z2nz/tot:.4f}   nonzero->0 {nz2z/tot:.4f}   [Prism 0.124 / 0.022]",flush=True)
print(f"  sign_flip_rate(nz)  {flip/max(1,nz_tr):.5f}   [Prism ~0.00002]",flush=True)
print(f"  scale_ratio tr/ptq  {(st_all.mean()/sp_all.mean()).item():.3f}   [Prism ~0.625]",flush=True)
print(f"  scale_std  tr {st_all.std().item():.4f}  ptq {sp_all.std().item():.4f}  ratio {(st_all.std()/sp_all.std()).item():.3f}  [Prism narrows]",flush=True)
print("\n=== weight delta by block (trained vs FP16) ===",flush=True)
for b in sorted(blk_delta):
    print(f"  block {b:2d}: {blk_delta[b]/blk_n[b]:.3f}",flush=True)

# islands
print("\n=== island deltas (trained vs FP16) ===",flush=True)
island_named=[(n,p) for n,p in tf.named_parameters() if is_island(n)]
for (n,p),tr in zip(island_named,island_values(ck,island_named)):
    base=p.data.float().to(dev); t=tr.float().to(dev); d=(t-base).norm()/base.norm().clamp_min(1e-9)
    tag=""
    if "proj_out" in n: tag=" <- Prism FROZEN (0%)"
    elif "norm_out" in n: tag=" <- Prism main compensator (100-150%)"
    elif "scale_shift_table" in n: tag=" <- Prism block-mod ~1-2%"
    print(f"  {n:55s} {d.item()*100:6.1f}%{tag}",flush=True)
print("\n[done]",flush=True)
