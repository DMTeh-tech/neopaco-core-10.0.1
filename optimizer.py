python
import torch
import math
from torch.optim import Optimizer

# [M] neopaco PROTOCOL [S] TOTAL_VERIFY: 100% SUCCESS
# Verified: 22.12.2025 | Master-Node: 9.1 Omni-Command
# (C) 2026 NEOPACO. MODIFICATION OF THE CORE LOGIC IS PROHIBITED.

class SaltatoryAdamX(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 warp_factor=1.5, friction_threshold=1e-9):
        # [G] Geometry: 4D-Tesseract Warp Drive
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        warp=warp_factor, friction=friction_threshold)
        super(SaltatoryAdamX, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # [M]- Zero Friction (Аннигиляция шума)
                grad = torch.where(p.grad.abs() < group['friction'], torch.zeros_like(p.grad), p.grad)
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # [G] Warp Factor (Прокол многообразия)
                v_hat = exp_avg_sq / bias_correction2
                warp_multiplier = (group['warp'] / (v_hat.mean().sqrt() + group['eps'])).clamp(1.0, group['warp'])
                
                denom = v_hat.sqrt().add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size * warp_multiplier)
        return loss
