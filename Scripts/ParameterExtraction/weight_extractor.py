import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class TCResNet8(nn.Module):
    """ TC-ResNet8 implementation """
    def __init__(self, k=False, n_mels=40):
        super(TCResNet8, self).__init__()

        self.n_labels = 12
        self.n_convs = 6
        self.n_maps = [16, 24, 24, 32, 32, 48, 48]
        self.n_residual_convs = 3
        self.residual_n_maps = [16, 24, 32, 48]

        # TC-ResNet8
        
        # First Convolution Layer
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=self.n_maps[0], kernel_size=(n_mels, 3), 
                                padding = (0,1), bias=False)

        # Residual Convolution Layers
        self.residual_convs = [nn.Conv2d(self.residual_n_maps[i], self.residual_n_maps[i+1], kernel_size=1, stride=2, bias=False) 
                        for i in range(self.n_residual_convs)]
        for i, residual_conv in enumerate(self.residual_convs):
            self.add_module("conv_residual{}".format(i), residual_conv)
            self.add_module("bn_residual{}".format(i), nn.BatchNorm2d(self.residual_n_maps[i+1], affine=True))

        # Main Convolution Layers
        self.convs = [nn.Conv2d(self.n_maps[i], self.n_maps[i+1], kernel_size=(1, 9), padding=(0, 4), stride=(2 if (i%2 == 0) else 1), bias=False) 
                        for i in range(self.n_convs)]
        for i, conv in enumerate(self.convs):
            self.add_module("conv{}".format(i + 1), conv)
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(self.n_maps[i+1], affine=True))

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
                x_res = F.relu(getattr(self, "bn_residual{}".format(int((i-2)/2)))(x_res)) 
                x = F.relu(getattr(self, "bn{}".format(i))(y) + x_res)
                old_x = x
            else:
                x = getattr(self, "bn{}".format(i))(y)
                x = F.relu(y)

        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        return self.output(x)
 
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

if __name__ == '__main__':
    model = TCResNet8()
    model.load("TCResNet8")
    conv0 = model.conv0.weight.data.numpy()
    conv1 = model.conv1.weight.data.numpy()
    conv2 = model.conv2.weight.data.numpy()
    conv3 = model.conv3.weight.data.numpy()
    conv4 = model.conv4.weight.data.numpy()
    conv5 = model.conv5.weight.data.numpy()
    conv6 = model.conv6.weight.data.numpy()
    bn1_w = model.bn1.weight.data.numpy()
    bn2_w = model.bn2.weight.data.numpy()
    bn3_w = model.bn3.weight.data.numpy()
    bn4_w = model.bn4.weight.data.numpy()
    bn5_w = model.bn5.weight.data.numpy()
    bn6_w = model.bn6.weight.data.numpy()
    bn1_b = model.bn1.bias.data.numpy()
    bn2_b = model.bn2.bias.data.numpy()
    bn3_b = model.bn3.bias.data.numpy()
    bn4_b = model.bn4.bias.data.numpy()
    bn5_b = model.bn5.bias.data.numpy()
    bn6_b = model.bn6.bias.data.numpy()
    conv_residual0 = model.conv_residual0.weight.data.numpy()
    conv_residual1 = model.conv_residual1.weight.data.numpy()
    conv_residual2 = model.conv_residual2.weight.data.numpy()
    bn_residual0_w = model.bn_residual0.weight.data.numpy()
    bn_residual1_w = model.bn_residual1.weight.data.numpy()
    bn_residual2_w = model.bn_residual2.weight.data.numpy()
    bn_residual0_b = model.bn_residual0.bias.data.numpy()
    bn_residual1_b = model.bn_residual1.bias.data.numpy()
    bn_residual2_b = model.bn_residual2.bias.data.numpy()
    fc = model.output.weight.data.numpy()
