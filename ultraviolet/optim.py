"""NorMuon (Violetto Alg. 1: Nesterov, Polar Express, neuron-wise second moment, cautious WD scaled by lr^2) for
hidden 2D matrices + AdamW groups with per-group LRs (Violetto Table 1), plus the warmup-flat-linear-decay schedule.

Single GPU: `NorMuon(params)`. DDP: gradients are already averaged by DDP; pass `distributed=True` to split the
orthogonalisation across ranks (rank r updates params r::world, then broadcasts) -- results are identical either way.
Orthogonalisation runs in fp32 (Turing has no bf16; matrices here are small) or bf16 when `ortho_dtype` says so.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

# Polar Express, 5 iterations, safety factor 2e-2, cushion 2 (modded-nanogpt coefficients).
POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.no_grad()
def polar_express(G: torch.Tensor, steps: int = 5, dtype=torch.float32) -> torch.Tensor:
    X = G.to(dtype)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True).float() * 1.02 + 1e-6).to(dtype)
    for a, b, c in POLAR_EXPRESS[:steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.mT
    return X


class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.018, momentum=0.95, beta2=0.9, weight_decay=1.2, shape_scale=True,
                 distributed=False, ortho_dtype=torch.float32):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.shape_scale = shape_scale
        self.distributed = distributed and dist.is_available() and dist.is_initialized()
        self.ortho_dtype = ortho_dtype

    @torch.no_grad()
    def _update(self, p, group):
        g = p.grad
        if g is None:
            return
        st = self.state[p]
        m, n = p.shape
        red = getattr(p, "_normuon_red_dim", -1 if m >= n else -2)
        if not st:
            st["buf"] = torch.zeros_like(p, dtype=torch.float32)
            st["nu"] = torch.zeros((m, 1) if red == -1 else (1, n), dtype=torch.float32, device=p.device)
        mu = group["momentum"]
        g = g.float()
        buf = st["buf"]
        buf.lerp_(g, 1 - mu)                    # B = mu B + (1-mu) G
        gt = g.lerp(buf, mu)                    # Nesterov: (1-mu) G + mu B
        U = polar_express(gt, dtype=self.ortho_dtype).float()
        s = U.square().mean(dim=red, keepdim=True)
        rho = U.norm()
        st["nu"].lerp_(s, 1 - group["beta2"])
        D = U * st["nu"].clamp_min(1e-10).rsqrt()
        D.mul_(rho / D.norm().clamp_min(1e-10))
        lr = group["lr"]
        eff = lr * (max(1.0, m / n) ** 0.5 if self.shape_scale else 1.0)
        wd = group["weight_decay"] * lr * lr
        if wd:
            mask = (D * p) > 0                  # cautious WD (strict)
            p.sub_(p * mask, alpha=wd)
        p.sub_(D.to(p.dtype), alpha=eff)

    @torch.no_grad()
    def step(self, closure=None):
        if not self.distributed:
            for group in self.param_groups:
                for p in group["params"]:
                    self._update(p, group)
            return
        rank, world = dist.get_rank(), dist.get_world_size()
        allp = [(p, gr) for gr in self.param_groups for p in gr["params"]]
        handles = []
        for i, (p, gr) in enumerate(allp):
            owner = i % world
            if owner == rank:
                self._update(p, gr)
            handles.append(dist.broadcast(p.data, src=owner, async_op=True))
        for h in handles:
            h.wait()


# ------------------------------------------------------------------------------------------- groups (Table 1)
ADAM_GROUPS = {
    #  name           lr      betas         wd_mult
    "embed":       (0.0019, (0.5, 0.95), 150.0),
    "value_embed": (0.14, (0.75, 0.95), 5.0),
    "scalars":     (0.04, (0.9, 0.99), 0.0),
    "resid":       (0.0016, (0.9, 0.95), 0.0),
    "post":        (0.0080, (0.9, 0.95), 0.0),
    "gates":       (0.0019, (0.9, 0.99), 0.0),
    "mudd":        (0.0019, (0.9, 0.95), 0.0),
    "conv":        (0.01, (0.9, 0.95), 0.0),    # short-conv taps (not in Violetto)
    "ssm":         (0.005, (0.9, 0.95), 0.0),   # GDN A_log / dt_bias (not in Violetto)
    # hashed bigram table (not in Violetto): untied, sparse, zero-init input table like the value embedding, so it
    # takes the value-embedding settings (modded-nanogpt gives its n-gram table the value embeds' lr / wd multipliers)
    "bigram":      (0.14, (0.75, 0.95), 5.0),
}
ADAM_WD_BASE = 0.005


def classify(name: str, p) -> str:
    if name.startswith("embed."):
        return "embed"
    if name.startswith("value_embed."):
        return "value_embed"
    if name.startswith("mudd."):
        return "mudd"
    if name.startswith("bigram_embed."):
        return "bigram"
    if name == "bigram_lambdas":
        return "scalars"
    if name == "bigram_proj":
        return "normuon"
    leaf = name.split(".")[-1]
    if leaf in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down", "w_in", "wgate"):
        return "normuon"
    if leaf in ("qkv_scale", "o_scale"):
        return "scalars"
    if leaf in ("resid_attn", "resid_mlp"):
        return "resid"
    if leaf in ("post_attn", "post_mlp"):
        return "post"
    if leaf == "w" and ".conv" in name:
        return "conv"
    if leaf in ("A_log", "dt_bias"):
        return "ssm"
    if leaf in ("ve_gate", "attn_gate", "xsa_alpha", "w_a", "w_b"):
        return "gates"
    raise ValueError(f"unclassified parameter {name}")


def build_optimizers(model, muon_lr=0.018, lr_mult=1.0, adam_lr_mult=1.0, muon_wd=1.2, distributed=False,
                     ortho_dtype=torch.float32, fused_adam=True):
    """Returns (normuon, adamw). Each param group carries 'base_lr' and 'wd_mult' for `apply_schedule`."""
    buckets: dict[str, list] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets.setdefault(classify(name, p), []).append(p)
    mu_params = buckets.pop("normuon")
    mu = NorMuon(mu_params, lr=muon_lr * lr_mult, weight_decay=muon_wd, distributed=distributed,
                 ortho_dtype=ortho_dtype)
    for g in mu.param_groups:
        g["base_lr"] = g["lr"]
    groups = []
    for k, ps in buckets.items():
        lr, betas, wdm = ADAM_GROUPS[k]
        lr = lr * lr_mult * adam_lr_mult
        groups.append(dict(params=ps, lr=lr, base_lr=lr, betas=betas, wd_mult=wdm, weight_decay=0.0, name=k))
    kw = {"fused": True} if fused_adam and mu_params[0].is_cuda else {}
    adam = torch.optim.AdamW(groups, eps=1e-10, **kw)
    return mu, adam


def lr_factor(step: int, total: int, warmup: int, decay_frac: float = 0.2, final: float = 0.0) -> float:
    """Linear warmup, flat, linear decay to `final` (fraction of peak) over the last `decay_frac` of training."""
    if step < warmup:
        return (step + 1) / warmup
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return 1.0
    return final + (1 - final) * max(0.0, (total - step) / max(1, total - d0))


def apply_schedule(mu: NorMuon, adam, step: int, total: int, warmup: int, decay_frac: float = 0.2,
                   mom_warmup: int = 300, mom0: float = 0.85, mom1: float = 0.95, final: float = 0.0) -> float:
    f = lr_factor(step, total, warmup, decay_frac, final)
    for g in mu.param_groups:
        g["lr"] = g["base_lr"] * f
        g["momentum"] = mom0 + (mom1 - mom0) * min(1.0, step / max(1, mom_warmup))
    for g in adam.param_groups:
        g["lr"] = g["base_lr"] * f
        # AdamW decays by lr * weight_decay; Violetto applies lr^2 * lambda * mult
        g["weight_decay"] = g["lr"] * ADAM_WD_BASE * g["wd_mult"]
    return f
