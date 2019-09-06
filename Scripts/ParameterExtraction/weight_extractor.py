import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def fixed_point_precision(module):
    minimum = module.min()
    maximum = module.max()
    print("Min:{}, Max:{}".format(minimum, maximum))

def q7_to_float(x):
    bin_str = bin(x)
    bin_str = bin_str[2:]
    # Padding with zeros
    n_bin_str = len(bin_str)
    bin_str = ((8-n_bin_str)*'0') + bin_str
    # Initializing result
    result = 0
    pos = True
    val = 0.5
    # Iterating over all 8 bits
    for i, bit in enumerate(bin_str):
        # Checking sign on bit 0
        if i == 0:
            if bit == '1':
                pos = False
        # Building number
        else:
            if bit == '1':
                result += val
            val /= 2.0
    # Modifying sign
    if pos == False:
        result = -result
    return result


def f32_to_q7(x):
    val = np.zeros_like(x)
    abs_x = np.abs(x)
    # Bit 6
    if abs_x >= 0.5:
        val = 0.5
        abs_x -= 0.5
    else:
        val = 0.0
    # Bit 5
    if abs_x >= 0.25:
        val += 0.25
        abs_x -= 0.25
    # Bit 4
    if abs_x >= 0.125:
        val += 0.125
        abs_x -= 0.125
    # Bit 3
    if abs_x >= 0.0625:
        val += 0.0625
        abs_x -= 0.0625
    # Bit 2
    if abs_x >= 0.03125:
        val += 0.03125
        abs_x -= 0.03125
    # Bit 1
    if abs_x >= 0.015625:
        val += 0.015625
        abs_x -= 0.015625
    # Bit 0 
    if abs_x >= 0.0078125:
        val += 0.0078125
        abs_x -= 0.0078125
    # Sign bit
    val = val*128
    if x < 0:
        val = -val
    # Return fixed-point value
    return val

v_f32_to_q7 = np.vectorize(f32_to_q7)



class TCResNet8(nn.Module):
    """ TC-ResNet8 implementation """
    def __init__(self, n_mels=40):
        super(TCResNet8, self).__init__()

        self.n_labels = 12
        self.n_convs = 6
        self.n_maps = [16, 24, 24, 32, 32, 48, 48]

        # First Convolution Layer
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=self.n_maps[0], kernel_size=(n_mels, 3), padding=(0,1), bias=False)

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
            # Save for first layer
            x = F.relu(y)

        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        return self.output(x)
 
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

def write_to_file(w_list):
    """ Writes weights to text file """
    file1 = open("weights.txt", "w")
    # Conv Layers
    for i in range(len(w_list)-2):
        data_string = "#define CONV{}_WT ".format(i)
        data_string += "{"
        # Number of channels
        for b in range(w_list[i].shape[1]):
            # Width
            for c in range(w_list[i].shape[2]):
                # Height
                for d in range(w_list[i].shape[3]):
                    # Number of output channels:
                    for a in range(w_list[i].shape[0]):
                        data_string += str(int(w_list[i][a, b, c, d]))
                        data_string += ","
        data_string = data_string[:-1]
        data_string += "}\n"
        file1.write(data_string)
    # FC Layer
    data_string = "#define IP1_WT "
    data_string += "{"
    for b in range(w_list[-2].shape[1]):
        for a in range(w_list[-2].shape[0]):
            data_string += str(int(w_list[-2][a, b]))
            data_string += ","
    data_string = data_string[:-1]
    data_string += "}\n"
    file1.write(data_string)
    # FC Bias
    data_string = "#define IP1_BIAS "
    data_string += "{"
    for a in range(w_list[-1].shape[0]):
        data_string += str(int(w_list[-1][a]))
        data_string += ","
    data_string = data_string[:-1]
    data_string += "}\n"
    file1.write(data_string)


    file1.close() 

if __name__ == '__main__':
    # Creating model
    model = TCResNet8()
    model.load("TCResNet8")
    # Extracting weights
    conv0 = model.conv0.weight.data.numpy()
    conv1 = model.conv1.weight.data.numpy()
    conv2 = model.conv2.weight.data.numpy()
    conv3 = model.conv3.weight.data.numpy()
    conv4 = model.conv4.weight.data.numpy()
    conv5 = model.conv5.weight.data.numpy()
    conv6 = model.conv6.weight.data.numpy()
    fc = model.output.weight.data.numpy()
    fc_bias = model.output.bias.data.numpy()
    # Converting weights from floating point to fixed point
    parameters = [conv0, conv1, conv2, conv3, conv4, conv5, conv6, fc, fc_bias]
    fixed_parameters = []
    for i, element in enumerate(parameters):
        if i == 8:
            fixed_parameters.append(v_f32_to_q7(element/(2**2))) # Bias will require an lshift of 2
        else:
            fixed_parameters.append(v_f32_to_q7(element))
    # Changing weights
    write_to_file(fixed_parameters)
