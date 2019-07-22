import torch
import torch.nn.functional as F
import quantize            as q

BINARIZED = True

class Conv2dQNN(torch.nn.Conv2d):
    """
    Convolution layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       H            = 1.0):

        
        self.H = H
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, x):
        Wq = q.int_nn(self.weight/self.H)*self.H
        return F.conv2d(x, Wq, self.bias, self.stride, self.padding, self.dilation, self.groups) 



