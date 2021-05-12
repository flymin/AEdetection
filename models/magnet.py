from typing import List, Tuple
import torch
import torch.nn as nn
from misc.kerasAPI import Conv2dActReg


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, in_channel: int, structure: List, v_noise=0.0,
                 activation="relu", model_dir="./defensive_models",
                 reg_method="L2", reg_strength=0.0):
        super(DenoisingAutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.model_dir = model_dir
        self.v_noise = v_noise
        self.downwards = nn.ModuleList()
        self.upwards = nn.ModuleList()
        last_c = self.in_channel
        for layer in structure:
            if isinstance(layer, int):
                self.downwards.append(
                    Conv2dActReg(
                        last_c, layer, 3, padding=1,
                        activation=activation,
                        reg_method=reg_method,
                        reg_strength=reg_strength))
                last_c = layer
            elif layer == "max":
                # here we must suppose this size is divisable by 2
                self.downwards.append(nn.MaxPool2d(2))
            elif layer == "average":
                self.downwards.append(nn.AvgPool2d(2))
            else:
                raise NotImplementedError()

        for layer in reversed(structure):
            if isinstance(layer, int):
                self.upwards.append(
                    Conv2dActReg(
                        last_c, layer, 3, padding=1,
                        activation=activation,
                        reg_method=reg_method,
                        reg_strength=reg_strength))
                last_c = layer   
            elif layer == "max" or layer == "average":
                self.upwards.append(nn.Upsample(scale_factor=(2, 2)))

        self.decoded = Conv2dActReg(last_c, self.in_channel, (3, 3),
                                    padding=1, activation="sigmoid",
                                    reg_method=reg_method,
                                    reg_strength=reg_strength)

    def forward(self, x):
        reg_loss = 0
        for mod in self.downwards:
            if isinstance(mod, Conv2dActReg):
                x, reg = mod(x)
                reg_loss += reg
            else:
                x = mod(x)
        for mod in self.upwards:
            if isinstance(mod, Conv2dActReg):
                x, reg = mod(x)
                reg_loss += reg
            else:
                x = mod(x)
        x, reg = self.decoded(x)
        reg_loss += reg
        return x, reg_loss