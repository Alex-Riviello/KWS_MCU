import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import utility as util

def apply_quant(x, nbits):
    x = x.floor()
    if nbits < 32:
        max_val = 2**(nbits-1)-1
        min_val = -2**(nbits-1)+1
        mask = x>max_val
        mask = mask.float()
        x    = max_val*mask + (1-mask)*x
        mask = x<min_val
        mask = mask.float()
        x    = min_val*mask + (1-mask)*x
    return x

class IntNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nbits=8):
        ctx.save_for_backward(x)
        return apply_quant(x, nbits)
    
    @staticmethod
    def backward(ctx, dx):
        return dx, None

int_nn = IntNN.apply
