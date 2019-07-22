import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import quantize as q
from layers import *


class TCResNet8(nn.Module):
    """ TC-ResNet8 implementation """
    def __init__(self, n_mels=40):
        super(TCResNet8, self).__init__()

        self.n_labels = 12
        self.n_convs = 6
        self.n_maps = [16, 24, 24, 32, 32, 48, 48]
        self.n_residual_convs = 3
        self.residual_n_maps = [16, 24, 32, 48]

        # TC-ResNet8
        
        # First Convolution Layer
        self.conv0 = Conv2dQNN(in_channels=1, out_channels=self.n_maps[0], kernel_size=(n_mels, 3), 
                                padding = (0,1), bias=False)

        # Residual Convolution Layers
        self.residual_convs = [Conv2dQNN(self.residual_n_maps[i], self.residual_n_maps[i+1], kernel_size=1, stride=2, bias=False) 
                        for i in range(self.n_residual_convs)]
        for i, residual_conv in enumerate(self.residual_convs):
            self.add_module("conv_residual{}".format(i), residual_conv)
            #self.add_module("bn_residual{}".format(i), nn.BatchNorm2d(self.residual_n_maps[i+1], affine=True))

        # Main Convolution Layers
        self.convs = [Conv2dQNN(self.n_maps[i], self.n_maps[i+1], kernel_size=(1, 9), padding=(0, 4), stride=(2 if (i%2 == 0) else 1), bias=False) 
                        for i in range(self.n_convs)]
        for i, conv in enumerate(self.convs):
            self.add_module("conv{}".format(i + 1), conv)
            #self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(self.n_maps[i+1], affine=True))
        # Output Layer
        self.output = nn.Linear(self.n_maps[-1], self.n_labels)
        
        
    def forward(self, x):
        
        x = x.unsqueeze(1)

        for i in range(self.n_convs + 1): 
            # Conv0 to 12
            y = getattr(self, "conv{}".format(i))(x)
            # Save for first layer
            if i == 0: 
                x = F.relu(y)
                old_x = x
            # Add residuals every 2 layers
            elif ((i > 0) and ((i % 2) == 0)):
                x_res = getattr(self, "conv_residual{}".format(int((i-2)/2)))(old_x)

                # x_res = F.relu(getattr(self, "bn_residual{}".format(int((i-2)/2)))(x_res)) 
                x_res = F.relu(x_res)

                # x = F.relu(getattr(self, "bn{}".format(i))(y) + x_res)
                x = F.relu(y + x_res)
                old_x = x
            else:
                # x = getattr(self, "bn{}".format(i))(y)
                x = F.relu(y)
            x = self.dropout(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        return self.output(x)
 
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


""" https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Quant_guide.md """