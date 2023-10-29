import torch
from torch import nn

def _logmap(y):
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
    return torch.clamp(arctanh(y_norm)*y/y_norm, -10, 10)

def _expmap(u):
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    return torch.tanh(u_norm)*u/u_norm


def _mobius_matvec(m, x):
    x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    mx = x @ m
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
    res_c = torch.tanh(mx_norm / x_norm * arctanh(x_norm)) * mx / mx_norm
    # cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    # res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    # res = torch.where(cond, res_0, res_c)
    return res_c


def arctanh(X):
    return Artanh.apply(X)

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
