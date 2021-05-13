import torch

def jsd(P, Q):
        _P = P / torch.norm(P, p=1)
        _Q = Q / torch.norm(Q, p=1)
        _M = 0.5 * (_P + _Q)
        entropy = lambda x, y: torch.sum(x * torch.log(x / y))
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))