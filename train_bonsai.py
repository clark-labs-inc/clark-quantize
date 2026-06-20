"""Bonsai QAT: full pipeline to reverse-engineer Prism's method on a tiny DiT.

Pipeline:
  1. Train FP16 teacher on CIFAR-10 (flow matching)
  2. Initialize student from teacher with optimal-scale ternary PTQ
  3. QAT: STE ternary forward + self-distillation + proximal pull
  4. Compare: teacher vs PTQ vs QAT (velocity MSE, FID-proxy, code stats)

Usage:
  python train_bonsai.py --phase all          # full pipeline
  python train_bonsai.py --phase teacher      # just train teacher
  python train_bonsai.py --phase qat          # QAT from saved teacher
  python train_bonsai.py --phase analyze      # analyze saved checkpoints
"""
from __future__ import annotations
import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

from bonsai_quant import (
    TernaryLinear, wrap_model_ternary, optimal_ternary_codes,
    ProximalPullScheduler, ForwardBlendScheduler,
    evaluate_ternary_error, get_all_code_stats, bake_codes, count_params,
    DEFAULT_ISLAND_KEYWORDS, is_island,
)
from tiny_dit import TinyDiT, FlowMatchScheduler


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size=128, n_workers=2, subset_size=None, root="./data"):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ])

    # Try torchvision CIFAR-10 first (only if already downloaded)
    cifar_path = os.path.join(root, "cifar-10-batches-py")
    if os.path.exists(cifar_path):
        try:
            train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
        except Exception as e:
            print(f"[data] CIFAR-10 load failed ({e}), using synthetic data")
            train_set = SyntheticDataset(n_samples=5000, img_size=32, transform=transform)
            test_set = SyntheticDataset(n_samples=1000, img_size=32, transform=transform)
    else:
        print("[data] CIFAR-10 not found, using synthetic colored-shapes dataset")
        train_set = SyntheticDataset(n_samples=5000, img_size=32, transform=transform)
        test_set = SyntheticDataset(n_samples=1000, img_size=32, transform=transform)

    if subset_size:
        train_set = Subset(train_set, list(range(min(subset_size, len(train_set)))))
        test_set = Subset(test_set, list(range(min(subset_size // 5, len(test_set)))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=n_workers, pin_memory=True)
    return train_loader, test_loader


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic image dataset: colored circles and squares on random backgrounds.

    For proof-of-concept when CIFAR-10 download is unavailable.
    """

    def __init__(self, n_samples=5000, img_size=32, transform=None):
        self.n_samples = n_samples
        self.img_size = img_size
        self.transform = transform
        # Pre-generate seeds for reproducibility
        self.seeds = torch.arange(n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(int(self.seeds[idx]))
        img = torch.randn(3, self.img_size, self.img_size, generator=g) * 0.3

        # Add a colored shape
        shape_type = idx % 4  # 0: circle, 1: square, 2: vertical bar, 3: horizontal bar
        color = torch.rand(3, generator=g) * 0.8 + 0.2  # [0.2, 1.0]
        cx, cy = torch.randint(8, 24, (2,), generator=g)
        size = torch.randint(4, 10, (1,), generator=g).item()

        if shape_type == 0:  # circle
            yy, xx = torch.meshgrid(torch.arange(self.img_size), torch.arange(self.img_size), indexing='ij')
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < size ** 2
        elif shape_type == 1:  # square
            mask = torch.zeros(self.img_size, self.img_size, dtype=torch.bool)
            mask[max(0, cy - size):cy + size, max(0, cx - size):cx + size] = True
        elif shape_type == 2:  # vertical bar
            mask = torch.zeros(self.img_size, self.img_size, dtype=torch.bool)
            mask[:, max(0, cx - 2):cx + 2] = True
        else:  # horizontal bar
            mask = torch.zeros(self.img_size, self.img_size, dtype=torch.bool)
            mask[max(0, cy - 2):cy + 2, :] = True

        for c in range(3):
            img[c][mask] = color[c]

        # Convert to PIL and back for transform compatibility
        from PIL import Image
        img_np = ((img * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)

        if self.transform:
            pil_img = self.transform(pil_img)
        else:
            pil_img = T.ToTensor()(pil_img)
            pil_img = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(pil_img)

        return pil_img, shape_type


# ---------------------------------------------------------------------------
# Training: FP16 teacher
# ---------------------------------------------------------------------------

def train_teacher(model, train_loader, device, epochs=20, lr=2e-4, save_path="checkpoints"):
    """Train FP16 teacher with flow matching on CIFAR-10."""
    os.makedirs(save_path, exist_ok=True)
    teacher_path = os.path.join(save_path, "teacher.pt")

    if os.path.exists(teacher_path):
        model.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        print(f"[teacher] loaded from {teacher_path}")
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    model = model.to(device)
    model.train()

    print(f"[teacher] training {epochs} epochs on CIFAR-10...")
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for x0, _ in train_loader:
            x0 = x0.to(device)
            B = x0.shape[0]
            noise = torch.randn_like(x0)
            t = torch.rand(B, device=device)  # uniform [0, 1]

            # Flow matching: x_t = (1-t)*noise + t*x0, target v = x0 - noise
            x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x0
            v_target = x0 - noise

            v_pred = model(x_t, t)
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}  time={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), teacher_path)
    print(f"[teacher] saved to {teacher_path}")
    return model


# ---------------------------------------------------------------------------
# QAT: Prism-style quantization-aware training
# ---------------------------------------------------------------------------

def train_qat(teacher, student, train_loader, device,
              steps=5000, lr_u=2e-5, lr_s=1e-4, lr_island=1e-6,
              prox_schedule=None, blend_steps=500,
              teacher_weight=1.0, velocity_weight=1.0,
              save_path="checkpoints"):
    """Prism-style QAT: STE ternary + self-distillation + proximal pull.

    Args:
        teacher: frozen FP16 model
        student: model with TernaryLinear-wrapped trunk
        train_loader: CIFAR-10 training data
        device: torch device
        steps: total training steps
        lr_u: learning rate for trunk master weights
        lr_s: learning rate for group scales (not used separately here — scales are derived from u)
        lr_island: learning rate for island parameters
        prox_schedule: list of (step, rho) pairs for proximal pull
        blend_steps: steps to ramp forward blend from 0 to 1
    """
    os.makedirs(save_path, exist_ok=True)

    # Freeze teacher
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = student.to(device)
    student.train()

    # Separate parameter groups
    trunk_params = []
    island_params = []
    for name, p in student.named_parameters():
        if isinstance(p, nn.Parameter) and p.requires_grad:
            if 'u' in name.split('.')[-1]:  # TernaryLinear master weight
                trunk_params.append(p)
            else:
                island_params.append(p)

    # Freeze proj_out completely (Prism's key finding)
    for name, mod in student.named_modules():
        if 'proj_out' in name and isinstance(mod, nn.Linear):
            mod.weight.requires_grad = False
            if mod.bias is not None:
                mod.bias.requires_grad = False
            print(f"  [qat] FROZEN: {name}")

    optimizer = torch.optim.AdamW([
        {'params': trunk_params, 'lr': lr_u},
        {'params': island_params, 'lr': lr_island},
    ], weight_decay=0.0)

    # Re-add trunk params that were accidentally classified
    # Actually, let's be more careful
    trunk_params_clean = []
    island_params_clean = []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        # Check if this belongs to a TernaryLinear
        parent_name = '.'.join(name.split('.')[:-1])
        parent_mod = student
        for part in parent_name.split('.'):
            if part:
                parent_mod = getattr(parent_mod, part, None)
                if parent_mod is None:
                    break
        if isinstance(parent_mod, TernaryLinear):
            trunk_params_clean.append(p)
        else:
            island_params_clean.append(p)

    optimizer = torch.optim.AdamW([
        {'params': trunk_params_clean, 'lr': lr_u},
        {'params': island_params_clean, 'lr': lr_island},
    ], weight_decay=0.0)

    prox_scheduler = ProximalPullScheduler(prox_schedule or [(0, 0.0), (blend_steps, 0.0),
                                                              (1000, 0.01), (3000, 0.05), (5000, 0.10)])
    blend_scheduler = ForwardBlendScheduler(blend_steps)

    # Training loop
    print(f"[qat] training {steps} steps  lr_u={lr_u}  lr_island={lr_island}  blend={blend_steps} steps")
    print(f"  trunk params: {sum(p.numel() for p in trunk_params_clean)/1e6:.2f}M")
    print(f"  island params: {sum(p.numel() for p in island_params_clean)/1e6:.2f}M")

    step = 0
    total_loss = 0
    total_distill = 0
    n_log = 100

    data_iter = iter(train_loader)

    while step < steps:
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x0, _ = next(data_iter)

        x0 = x0.to(device)
        B = x0.shape[0]
        noise = torch.randn_like(x0)
        t = torch.rand(B, device=device)

        x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x0
        v_target = x0 - noise

        # Teacher prediction (no grad)
        with torch.no_grad():
            v_teacher = teacher(x_t, t)

        # Student prediction (quantized forward via STE)
        v_student = student(x_t, t)

        # Losses
        loss_target = F.mse_loss(v_student, v_target)
        loss_distill = F.mse_loss(v_student, v_teacher)
        loss = velocity_weight * loss_target + teacher_weight * loss_distill

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        optimizer.step()

        # Proximal pull after optimizer step
        rho = prox_scheduler.get_rho(step)
        if rho > 0:
            for name, mod in student.named_modules():
                if isinstance(mod, TernaryLinear):
                    mod.proximal_pull(rho)

        total_loss += loss_target.item()
        total_distill += loss_distill.item()
        step += 1

        if step % n_log == 0:
            avg_target = total_loss / n_log
            avg_distill = total_distill / n_log
            alpha = blend_scheduler.get_alpha(step)
            print(f"  step {step:5d}/{steps}  target={avg_target:.5f}  distill={avg_distill:.5f}  "
                  f"rho={rho:.4f}  alpha={alpha:.2f}")
            total_loss = 0
            total_distill = 0

    # Save student
    student_path = os.path.join(save_path, "student_qat.pt")
    torch.save(student.state_dict(), student_path)
    print(f"[qat] saved to {student_path}")

    return student


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, test_loader, device, n_batches=10):
    """Evaluate velocity MSE on test data."""
    model = model.to(device)
    model.eval()
    total_mse = 0
    count = 0
    for i, (x0, _) in enumerate(test_loader):
        if i >= n_batches:
            break
        x0 = x0.to(device)
        B = x0.shape[0]
        noise = torch.randn_like(x0)
        t = torch.rand(B, device=device)
        x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x0
        v_target = x0 - noise
        v_pred = model(x_t, t)
        total_mse += F.mse_loss(v_pred, v_target).item() * B
        count += B
    return total_mse / max(count, 1)


@torch.no_grad()
def evaluate_trajectory(model, test_loader, device, n_samples=50, num_steps=8):
    """Multi-step trajectory error: how far does the rollout diverge?"""
    model = model.to(device)
    model.eval()
    total_err = 0
    count = 0

    for i, (x0, _) in enumerate(test_loader):
        if i * x0.shape[0] >= n_samples:
            break
        x0 = x0.to(device)
        B = x0.shape[0]
        noise = torch.randn_like(x0)

        # Generate from noise using the model
        dt_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        x = noise.clone()

        for j in range(num_steps):
            t = torch.full((B,), dt_vals[j], device=device)
            v = model(x, t)
            x = x + v * (dt_vals[j] - dt_vals[j + 1])

        # Compare generated samples to real images (proxy for trajectory error)
        total_err += F.mse_loss(x, x0).item() * B
        count += B

    return total_err / max(count, 1)


@torch.no_grad()
def generate_samples(model, device, n=16, num_steps=20, img_size=32):
    """Generate images from the model."""
    model = model.to(device)
    model.eval()
    dt_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    x = torch.randn(n, 3, img_size, img_size, device=device)

    for j in range(num_steps):
        t = torch.full((n,), dt_vals[j], device=device)
        v = model(x, t)
        x = x + v * (dt_vals[j] - dt_vals[j + 1])

    # Denormalize from [-1, 1] to [0, 1]
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_image_grid(images, path, nrow=4):
    """Save a grid of images."""
    import torchvision.utils as vutils
    vutils.save_image(images, path, nrow=nrow, normalize=False)


# ---------------------------------------------------------------------------
# PTQ baseline (no training, just optimal-scale ternary)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_ptq_ternary(model, group=128):
    """Apply optimal-scale ternary PTQ to a model (in-place, fake quantization)."""
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if is_island(name):
            continue
        W = mod.weight.data.float()
        t, s = optimal_ternary_codes(W, group)
        # Dequantize: Wq = t * s (broadcast s across input dim)
        if W.dim() == 4:
            out, inp, kh, kw = W.shape
            Wq = (t.reshape(out, inp, kh, kw) * s.unsqueeze(-1).unsqueeze(-1))
        else:
            Wq = t * s
        mod.weight.data.copy_(Wq.to(mod.weight.dtype))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "teacher", "qat", "analyze", "sample"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--teacher-epochs", type=int, default=20)
    ap.add_argument("--qat-steps", type=int, default=5000)
    ap.add_argument("--lr-u", type=float, default=2e-5)
    ap.add_argument("--lr-island", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--subset", type=int, default=None,
                    help="Use only N training images (for quick testing)")
    ap.add_argument("--save", default="checkpoints")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, subset_size=args.subset)

    # Model config
    model_kwargs = dict(img_size=32, patch_size=2, in_channels=3,
                        dim=args.dim, depth=args.depth, num_heads=args.heads)

    # ---- Phase: teacher ----
    if args.phase in ("all", "teacher"):
        print("\n" + "=" * 70)
        print("PHASE 1: Train FP16 teacher")
        print("=" * 70)
        teacher = TinyDiT(**model_kwargs)
        teacher = train_teacher(teacher, train_loader, device,
                                epochs=args.teacher_epochs, save_path=args.save)

    # ---- Phase: QAT ----
    if args.phase in ("all", "qat"):
        print("\n" + "=" * 70)
        print("PHASE 2: Prism-style QAT")
        print("=" * 70)

        # Load teacher
        teacher = TinyDiT(**model_kwargs).to(device)
        teacher_path = os.path.join(args.save, "teacher.pt")
        teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
        teacher.eval()

        # Create student: copy teacher, then wrap trunk with TernaryLinear
        student = TinyDiT(**model_kwargs)
        student.load_state_dict(teacher.state_dict())

        n_ternary, n_island, _ = wrap_model_ternary(student, group=args.group)

        # PTQ baseline: evaluate before training
        print("\n[qat] evaluating PTQ baseline (before QAT)...")
        ptq_vel = evaluate_model(student, test_loader, device)
        ptq_traj = evaluate_trajectory(student, test_loader, device, n_samples=50)
        ptq_codes = get_all_code_stats(student)
        print(f"  velocity MSE: {ptq_vel:.6f}")
        print(f"  trajectory MSE: {ptq_traj:.6f}")
        print(f"  code stats: zero_frac={ptq_codes['zero_frac']*100:.1f}%  "
              f"same={ptq_codes['same']*100:.1f}%  sign_flip={ptq_codes['sign_flip']*100:.4f}%")

        # QAT
        student = train_qat(teacher, student, train_loader, device,
                           steps=args.qat_steps, lr_u=args.lr_u, lr_island=args.lr_island,
                           save_path=args.save)

        # Evaluate after QAT
        print("\n[qat] evaluating after QAT...")
        qat_vel = evaluate_model(student, test_loader, device)
        qat_traj = evaluate_trajectory(student, test_loader, device, n_samples=50)
        qat_codes = get_all_code_stats(student)
        print(f"  velocity MSE: {qat_vel:.6f}")
        print(f"  trajectory MSE: {qat_traj:.6f}")
        print(f"  code stats: zero_frac={qat_codes['zero_frac']*100:.1f}%  "
              f"same={qat_codes['same']*100:.1f}%  sign_flip={qat_codes['sign_flip']*100:.4f}%  "
              f"activated={qat_codes['activated']*100:.1f}%  deactivated={qat_codes['deactivated']*100:.1f}%")

        # Save comparison
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        teacher_vel = evaluate_model(teacher, test_loader, device)
        teacher_traj = evaluate_trajectory(teacher, test_loader, device, n_samples=50)
        print(f"  {'model':20s} {'vel_mse':>12s} {'traj_mse':>12s} {'zero_frac':>10s} {'sign_flip':>10s}")
        print(f"  {'teacher (FP16)':20s} {teacher_vel:>12.6f} {teacher_traj:>12.6f} {'--':>10s} {'--':>10s}")
        print(f"  {'PTQ ternary':20s} {ptq_vel:>12.6f} {ptq_traj:>12.6f} {ptq_codes['zero_frac']*100:>9.1f}% {ptq_codes['sign_flip']*100:>9.4f}%")
        print(f"  {'QAT ternary':20s} {qat_vel:>12.6f} {qat_traj:>12.6f} {qat_codes['zero_frac']*100:>9.1f}% {qat_codes['sign_flip']*100:>9.4f}%")
        print(f"\n  QAT velocity improvement: {(1 - qat_vel/ptq_vel)*100:.1f}%")
        print(f"  QAT trajectory improvement: {(1 - qat_traj/ptq_traj)*100:.1f}%")

        # Generate samples from each
        print("\n[sample] generating images...")
        teacher_imgs = generate_samples(teacher, device, n=16)
        save_image_grid(teacher_imgs, os.path.join(args.out, "teacher_samples.png"))

        # PTQ samples (need a fresh copy since we modified in-place)
        ptq_model = TinyDiT(**model_kwargs)
        ptq_model.load_state_dict(teacher.state_dict())
        apply_ptq_ternary(ptq_model, group=args.group)
        ptq_model = ptq_model.to(device)
        ptq_imgs = generate_samples(ptq_model, device, n=16)
        save_image_grid(ptq_imgs, os.path.join(args.out, "ptq_samples.png"))

        qat_imgs = generate_samples(student, device, n=16)
        save_image_grid(qat_imgs, os.path.join(args.out, "qat_samples.png"))

        print(f"  saved to {args.out}/teacher_samples.png, ptq_samples.png, qat_samples.png")

    # ---- Phase: analyze ----
    if args.phase == "analyze":
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)

        teacher = TinyDiT(**model_kwargs).to(device)
        teacher.load_state_dict(torch.load(os.path.join(args.save, "teacher.pt"),
                                           map_location=device, weights_only=True))

        student = TinyDiT(**model_kwargs)
        student.load_state_dict(teacher.state_dict())
        wrap_model_ternary(student, group=args.group)
        student.load_state_dict(torch.load(os.path.join(args.save, "student_qat.pt"),
                                           map_location=device, weights_only=True))
        student = student.to(device)

        # Code stats
        codes = get_all_code_stats(student)
        print(f"\nCode transition analysis (init PTQ -> trained QAT):")
        print(f"  same codes:      {codes['same']*100:.2f}%")
        print(f"  sign flips:      {codes['sign_flip']*100:.4f}%")
        print(f"  activated:       {codes['activated']*100:.2f}%  (0 -> ±1)")
        print(f"  deactivated:     {codes['deactivated']*100:.2f}%  (±1 -> 0)")
        print(f"  init zero frac:  {codes['init_zero_frac']*100:.1f}%")
        print(f"  final zero frac: {codes['zero_frac']*100:.1f}%")

        # Island delta analysis
        print(f"\nIsland weight deltas (from teacher):")
        teacher_state = teacher.state_dict()
        student_state = student.state_dict()
        for name, p in student.named_parameters():
            if 'u' not in name.split('.')[-1]:  # island param
                if name in teacher_state:
                    delta = ((p.data - teacher_state[name].to(device)).norm() /
                             teacher_state[name].norm().clamp_min(1e-9)).item()
                    if delta > 0.001:
                        print(f"  {name:50s} delta={delta*100:.2f}%")

        # Param budget
        info = count_params(student)
        print(f"\nParameter budget:")
        print(f"  trunk: {info['trunk_params']/1e6:.2f}M ({info['trunk_pct']:.1f}%)")
        print(f"  island: {info['island_params']/1e6:.2f}M ({info['island_pct']:.1f}%)")
        print(f"  effective bpw: {info['effective_bpw']:.3f}")
        print(f"  compression: {info['compression']:.2f}x")

    # ---- Phase: sample ----
    if args.phase == "sample":
        for tag, ckpt, wrap in [("teacher", "teacher.pt", False),
                                 ("ptq", "teacher.pt", True),
                                 ("qat", "student_qat.pt", True)]:
            model = TinyDiT(**model_kwargs)
            model.load_state_dict(torch.load(os.path.join(args.save, ckpt),
                                             map_location=device, weights_only=True))
            if wrap:
                if tag == "ptq":
                    apply_ptq_ternary(model, group=args.group)
                else:
                    # Re-wrap and load
                    wrap_model_ternary(model, group=args.group, verbose=False)
                    model.load_state_dict(torch.load(os.path.join(args.save, ckpt),
                                                     map_location=device, weights_only=True))
            model = model.to(device)
            imgs = generate_samples(model, device, n=16)
            save_image_grid(imgs, os.path.join(args.out, f"{tag}_samples.png"))
            print(f"saved {tag}_samples.png")


if __name__ == "__main__":
    main()
