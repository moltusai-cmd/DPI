import torch
import math
from torch.optim import Optimizer

class SpectreMuon(Optimizer):
    """
    Spectre-Muon v1.1: Merging Muon (Newton-Schulz) with DSO (Iron Anchor).
    - Muon: Orthogonalizes updates for 2D matrices.
    - DSO: Anchors weights to DPI initialization to prevent manifold drift.
    [v1.1 Fix]: Fallback to AdamW instead of SGD for scalars/embeddings to prevent NaNs at high HR.
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, anchor_factor=0.5, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay, 
                        anchor_factor=anchor_factor, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            af = group['anchor_factor']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                # Heuristique pour isoler les embeddings : on limite Muon aux tenseurs 2D
                # qui n'ont pas des dimensions trop déséquilibrées (ratio < 10).
                # (ex: vocab=16384, dim=768 donne un ratio ~21 -> sera traité via AdamW)
                is_2d = p.ndim >= 2
                ratio = max(p.shape) / min(p.shape) if is_2d else 0
                use_muon = is_2d and (ratio < 10)

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)
                    if not use_muon:
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1

                # --- 1. IRON ANCHOR (DSO Logic) ---
                if wd != 0 and hasattr(p, 'dpi_anchor'):
                    recall = lr * wd * af
                    p.data.lerp_(p.dpi_anchor, recall)
                elif wd != 0 and not use_muon:
                    # Standard weight decay for Adam params
                    p.data.mul_(1 - lr * wd)

                # --- 2. UPDATE LOGIC ---
                if use_muon:
                    # Logique Muon Newton-Schulz
                    buf = state['momentum']
                    buf.mul_(momentum).add_(grad)
                    
                    g = buf
                    # [Correction v1.2] Le vrai Muon utilise le Ratio des dimensions et non la taille absolue !
                    # scale = sqrt(3072) donnait 55.4 (gigantesque). scale = sqrt(3072/768) donne 2.0 (correct).
                    scale = max(1, g.shape[0] / g.shape[1])**0.5
                    
                    # On optimise les multiplications de matrices rectangulaires
                    if g.shape[0] > g.shape[1]:
                        X = g / (torch.linalg.norm(g) + 1e-7)
                        for _ in range(ns_steps):
                            X = 1.5 * X - 0.5 * X @ (X.t() @ X)
                    else:
                        X = g / (torch.linalg.norm(g) + 1e-7)
                        for _ in range(ns_steps):
                            X = 1.5 * X - 0.5 * (X @ X.t()) @ X
                    
                    p.data.add_(X, alpha=-lr * scale)
                else:
                    # Vrai AdamW pour les gains, biais et embeddings (évite le NaN à lr=1e-3)
                    buf = state['momentum']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    buf.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    
                    p.addcdiv_(buf, denom, value=-step_size)


class DPISpectralOptimizer(Optimizer):
    """
    DPI Spectral Optimizer (DSO) v1.4 - "Iron Anchor" Edition (GOLD STANDARD)
    
    Verified Record: 5.23 Val Loss @ 8e-4 LR with 15% Noise.
    This version focuses on positional stability to allow extreme compression
    without losing the geometric foundation.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, anchor_factor=2.0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        anchor_factor=anchor_factor, amsgrad=amsgrad)
        super(DPISpectralOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                beta1, beta2 = group['betas']

                # 1. Iron Anchor Decay
                if group['weight_decay'] != 0 and hasattr(p, 'dpi_anchor'):
                    decay_force = group['lr'] * group['weight_decay']
                    total_recall = decay_force * group['anchor_factor']
                    p.data.mul_(1 - total_recall)
                    p.data.add_(p.dpi_anchor, alpha=total_recall)

                # 2. AdamW Core
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
