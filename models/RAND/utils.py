import torch
from torch.nn import functional as F
import random

def FFT_for_Period(inputs, m):
    xf = torch.fft.rfft(inputs, dim=1)
    frequency_list = abs(xf)
    _, f = torch.topk(frequency_list, dim=1, k=m)
    f += 1
    return torch.ceil(xf.shape[1] / f), f

def avgpool(x_r, p):
    """
    args:
        x_r: shape = (b, T, d)
        p: shape = (b, m, d)
    """
    res = []
    b, m, d = p.shape
    x_r = x_r.permute(0, 2, 1)
    p = p.permute(0, 2, 1)
    res = []
    for batch_idx in range(b):
        rd = []
        for d_idx in range(d):
            xi = x_r[batch_idx, d_idx, :].view(1, -1)
            pi = p[batch_idx, d_idx, :]
            rm = []
            for kernel in pi:
                x = F.pad(xi, (kernel.item()-1, 0), mode="replicate")
                x = F.avg_pool1d(x, kernel.item(), 1, 0).view(-1)
                rm.append(x)

            rm = torch.stack(rm)   # (m, T)
            rd.append(rm)

        rd = torch.stack(rd)   # (d, m, T)
        res.append(rd)

    res = torch.stack(res)   # (b, d, m, T)
    res = res.permute(0, 2, 3, 1)   # (b, m, T, d)
    return res

def sample(train_set, Ns=100):
    """
    args:
        train_set: shape = (n, T, m)
    """
    n = train_set.shape[0]
    index = torch.randint(0, n, (Ns,)).to(train_set.device)
    return train_set[index,...]

if __name__ == "__main__":
    x_r = torch.randn((32, 336, 8))
    p = torch.randint(1, 170, (32, 100, 8))
    print(avgpool(x_r, p).shape)