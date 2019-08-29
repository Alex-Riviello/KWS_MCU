import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class CNN6(nn.Module):
    """ TC-ResNet8 implementation """
    def __init__(self, n_mels=40):
        super(CNN6, self).__init__()

        self.n_labels = 12
        self.n_convs = 6
        self.n_maps = [16, 24, 24, 32, 32, 48, 48]

        # TC-ResNet8
        
        # First Convolution Layer
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=self.n_maps[0], kernel_size=(n_mels, 3), 
                                padding = (0,1), bias=False)

        # Main Convolution Layers
        self.convs = [nn.Conv2d(self.n_maps[i], self.n_maps[i+1], kernel_size=(1, 9), padding=(0, 4), stride=(2 if (i%2 == 0) else 1), bias=False) 
                        for i in range(self.n_convs)]
        for i, conv in enumerate(self.convs):
            self.add_module("conv{}".format(i + 1), conv)
        # Output Layer
        self.output = nn.Linear(self.n_maps[-1], self.n_labels)
        
        
    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = x[:, :, :, :-1]

        for i in range(self.n_convs + 1): 
            # Conv0 to 12
            y = getattr(self, "conv{}".format(i))(x)
            x = F.relu(y)

        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        return self.output(x)
 
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

if __name__ == '__main__':
    model = CNN6()
    model.load("CNN6")
    conv0 = model.conv0.weight.data.numpy()
    conv1 = model.conv1.weight.data.numpy()
    conv2 = model.conv2.weight.data.numpy()
    conv3 = model.conv3.weight.data.numpy()
    conv4 = model.conv4.weight.data.numpy()
    conv5 = model.conv5.weight.data.numpy()
    conv6 = model.conv6.weight.data.numpy()
    fc = model.output.weight.data.numpy()
