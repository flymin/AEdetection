import torch
import torch.nn as nn

class L2Reg(nn.Module):
    def __init__(self, reg_strength=0.0):
        super(L2Reg, self).__init__()
        self.reg_strength = reg_strength

    def forward(self, x):
        return self.reg_strength * torch.square(x).sum()


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.normal(0., self.stddev, x.size()).to(x.device)
            return noise + x
        else:
            return x


class Conv2dActReg(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, kernel=3, padding=1,
                 activation="sig", reg_method="L2", reg_strength=0.0):
        super(Conv2dActReg, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel,
                              padding=padding, stride=1)
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise NotImplementedError()
        if reg_method == "L2":
            self.reg = L2Reg(reg_strength)
        elif reg_method == "noise":
            self.reg = GaussianNoise(reg_strength)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        if self.training and isinstance(self.reg, GaussianNoise):
            x = x + self.reg(x)
            return x, 0
        elif isinstance(self.reg, GaussianNoise):
            return x, 0
        else:
            return x, self.reg(x)
