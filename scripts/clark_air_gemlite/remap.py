"""diffusers -> NVlabs-reference key remap for Clark Air Sana, with QKV/KV fusion.

ComfyUI's native Sana (via ComfyUI_ExtraModels) builds `SanaMS` with the *reference*
layout (fused `attn.qkv` / `cross_attn.kv_linear`, `mlp.{inverted,depth,point}_conv.conv`,
`x_embedder`/`t_embedder`/`t_block`/`y_embedder`/`final_layer` islands). Our GemLite pack is
keyed in *diffusers* layout. This module rewrites our pack into the reference layout so the
ComfyUI loader node can drop the GemLite trunk straight into `SanaMS`.

Verified against a fresh `SanaMS(**SanaMS_1600M_P1_D20)` skeleton (318 tensors). Biases:
`attn.qkv` and `mlp.point_conv.conv` carry none; everything else matches our pack.
"""
import numpy as np
import torch


def unpack2b(packed, out, inn):  # uint8 [out, in/4] -> ternary int8 [out, in]
    b = packed.astype(np.uint8); lev = np.empty((out, inn), np.int8)
    lev[:, 0::4] = b & 3; lev[:, 1::4] = (b >> 2) & 3; lev[:, 2::4] = (b >> 4) & 3; lev[:, 3::4] = (b >> 6) & 3
    return lev.astype(np.int8) - 1


def pack2b(code):  # int8/float [out,in] in {-1,0,1} -> uint8 [out, in/4]
    lev = (code + 1).astype(np.uint8)
    return (lev[:, 0::4] | (lev[:, 1::4] << 2) | (lev[:, 2::4] << 4) | (lev[:, 3::4] << 6)).astype(np.uint8)


def _block_trunk_map(n):
    """(reference module, diffusers sources to concat along out-dim)."""
    p, b = f"transformer_blocks.{n}.", f"blocks.{n}."
    return [
        (b + "attn.qkv",             [p + "attn1.to_q", p + "attn1.to_k", p + "attn1.to_v"]),
        (b + "attn.proj",            [p + "attn1.to_out.0"]),
        (b + "cross_attn.q_linear",  [p + "attn2.to_q"]),
        (b + "cross_attn.kv_linear", [p + "attn2.to_k", p + "attn2.to_v"]),
        (b + "cross_attn.proj",      [p + "attn2.to_out.0"]),
        (b + "mlp.inverted_conv.conv", [p + "ff.conv_inverted"]),
        (b + "mlp.point_conv.conv",  [p + "ff.conv_point"]),
    ]


def _block_island_map(n):
    p, b = f"transformer_blocks.{n}.", f"blocks.{n}."
    return {
        p + "ff.conv_depth.weight": b + "mlp.depth_conv.conv.weight",
        p + "ff.conv_depth.bias":   b + "mlp.depth_conv.conv.bias",
        p + "scale_shift_table":    b + "scale_shift_table",
    }


TOP_ISLAND_MAP = {
    "patch_embed.proj.weight": "x_embedder.proj.weight",
    "patch_embed.proj.bias": "x_embedder.proj.bias",
    "time_embed.emb.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_embed.emb.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_embed.emb.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "time_embed.emb.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_embed.linear.weight": "t_block.1.weight",
    "time_embed.linear.bias": "t_block.1.bias",
    "caption_projection.linear_1.weight": "y_embedder.y_proj.fc1.weight",
    "caption_projection.linear_1.bias": "y_embedder.y_proj.fc1.bias",
    "caption_projection.linear_2.weight": "y_embedder.y_proj.fc2.weight",
    "caption_projection.linear_2.bias": "y_embedder.y_proj.fc2.bias",
    "caption_norm.weight": "attention_y_norm.weight",
    "proj_out.weight": "final_layer.linear.weight",
    "proj_out.bias": "final_layer.linear.bias",
    "scale_shift_table": "final_layer.scale_shift_table",
}


def _island_store(t):
    """Large island weight matrices -> FP8 (e4m3) to shrink the pack; small/sensitive
    tensors (norms, scale_shift_table, biases) stay bf16. The loader converts back to bf16,
    so no per-key metadata is needed (safetensors preserves the float8 dtype + shape)."""
    if t.dim() >= 2 and t.numel() > 100_000:
        return t.to(torch.float8_e4m3fn).contiguous()
    return t.to(torch.bfloat16).contiguous()


def reference_quant_modules(num_layers=20):
    """Reference module names that become GemLite layers (used by the loader's swap loop)."""
    names = []
    for n in range(num_layers):
        names += [ref for ref, _ in _block_trunk_map(n)]
    return names


def build_reference_pack(sd, num_layers=20):
    """Rewrite our diffusers-keyed GemLite pack `sd` into the reference layout.

    Trunk layers keep the `<ref>::codes2b/::scale/::shape` schema (q/k/v and k/v fused along
    the out dim); biases and all islands are renamed bf16 tensors. Returns the new dict.
    """
    out = {}
    for n in range(num_layers):
        for ref, srcs in _block_trunk_map(n):
            # fuse codes (unpack -> concat out-dim -> repack) + scales + bias
            levels, scales, biases, inn = [], [], [], None
            for s in srcs:
                o, i = [int(x) for x in sd[s + "::shape"].tolist()]
                inn = i if inn is None else inn
                assert i == inn, f"{ref}: in_features mismatch {i} vs {inn}"
                levels.append(unpack2b(sd[s + "::codes2b"].numpy(), o, i))
                scales.append(sd[s + "::scale"])
                if s + ".bias" in sd:
                    biases.append(sd[s + ".bias"])
            fused = np.concatenate(levels, axis=0)
            out[ref + "::shape"] = torch.tensor([fused.shape[0], inn], dtype=torch.int32)
            out[ref + "::codes2b"] = torch.from_numpy(pack2b(fused))
            out[ref + "::scale"] = torch.cat(scales, dim=0).contiguous()
            if biases:
                assert len(biases) == len(srcs), f"{ref}: partial bias coverage"
                out[ref + ".bias"] = torch.cat(biases, dim=0).to(torch.bfloat16).contiguous()
        for d, r in _block_island_map(n).items():
            out[r] = _island_store(sd[d])
    for d, r in TOP_ISLAND_MAP.items():
        out[r] = _island_store(sd[d])
    return out
