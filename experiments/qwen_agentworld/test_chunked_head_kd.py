from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.qwen_agentworld.train_all2d_ternary_qat import (
    ChunkedTopKOtherHeadKL,
    alpha_quantized_weight,
)
from experiments.qwen_agentworld.train_mlp_triplet_survival_qat import topk_teacher_kl


def compare_case(*, mode: str, group_size: int, alpha: float) -> None:
    torch.manual_seed(17)
    tokens = 7
    dim = 16
    vocab = 53
    topk = 8
    chunk_rows = 11
    hidden = torch.randn(tokens, dim, dtype=torch.float32, requires_grad=True)
    weight = (0.02 * torch.randn(vocab, dim, dtype=torch.float32)).requires_grad_(True)
    teacher_logits = torch.randn(tokens, vocab, dtype=torch.float32)
    teacher_values, teacher_indices = teacher_logits.topk(topk, dim=-1)
    teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1)
    format_ids = teacher_indices[:2, :2].reshape(-1).unique()
    boost = 1.7

    if mode == "none":
        naive_weight = weight
    else:
        naive_weight = alpha_quantized_weight(weight, mode=mode, group_size=group_size, alpha=alpha)
    naive_logits = hidden.matmul(naive_weight.t()).unsqueeze(0)
    naive_loss, naive_stats = topk_teacher_kl(
        naive_logits,
        teacher_values,
        teacher_indices,
        temperature=1.0,
        format_ids=format_ids,
        format_boost=boost,
        teacher_logsumexp=teacher_logsumexp,
        use_other_bucket=True,
    )
    naive_loss.backward()
    naive_hidden_grad = hidden.grad.detach().clone()
    naive_weight_grad = weight.grad.detach().clone()

    hidden.grad = None
    weight.grad = None
    chunked_loss, stats_tensor = ChunkedTopKOtherHeadKL.apply(
        hidden,
        weight,
        teacher_values - teacher_logsumexp.reshape(-1, 1),
        teacher_indices,
        format_ids,
        mode,
        group_size,
        alpha,
        chunk_rows,
        boost,
    )
    chunked_loss.backward()
    chunked_hidden_grad = hidden.grad.detach().clone()
    chunked_weight_grad = weight.grad.detach().clone()

    loss_err = abs(float(chunked_loss.detach()) - float(naive_loss.detach()))
    hidden_err = (chunked_hidden_grad - naive_hidden_grad).norm() / naive_hidden_grad.norm().clamp_min(1e-12)
    weight_err = (chunked_weight_grad - naive_weight_grad).norm() / naive_weight_grad.norm().clamp_min(1e-12)
    stats = [float(x) for x in stats_tensor]
    assert loss_err < 1e-5, (mode, "loss", loss_err, float(chunked_loss), float(naive_loss))
    assert float(hidden_err) < 1e-4, (mode, "hidden_grad", float(hidden_err))
    assert float(weight_err) < 1e-4, (mode, "weight_grad", float(weight_err))
    assert math.isclose(stats[0], naive_stats["kl"], rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(stats[3], naive_stats["top1"], rel_tol=1e-5, abs_tol=1e-5)
    print(
        {
            "mode": mode,
            "loss": float(chunked_loss.detach()),
            "loss_err": loss_err,
            "hidden_grad_rel_err": float(hidden_err),
            "weight_grad_rel_err": float(weight_err),
        }
    )


def main() -> None:
    compare_case(mode="none", group_size=1, alpha=1.0)
    compare_case(mode="ternary", group_size=8, alpha=1.0)
    compare_case(mode="ternary", group_size=8, alpha=0.35)


if __name__ == "__main__":
    main()
