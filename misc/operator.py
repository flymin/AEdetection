import torch
import numpy as np


def entropy(x, y):
    return torch.sum(x * torch.log(x / y))


def jsd(P, Q):
    _P = P / torch.norm(P, p=1)
    _Q = Q / torch.norm(Q, p=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def l1_dist(x1, x2): return torch.sum(torch.abs(x1 - x2), dim=1)
def l2_dist(x1, x2): return torch.sum((x1 - x2)**2, dim=1)**.5


def kl(x1, x2):
    # Note: KL-divergence is not symentric.
    # Designed for probability distribution (e.g. softmax output).
    assert x1.shape == x2.shape
    # x1_2d, x2_2d = reshape_2d(x1), reshape_2d(x2)
    # Transpose to [?, #num_examples]
    x1_2d_t = x1.transpose()
    x2_2d_t = x2.transpose()
    # pdb.set_trace()
    e = entropy(x1_2d_t, x2_2d_t)
    e = torch.where(e == np.inf, torch.tensor(2.), e)
    return e
