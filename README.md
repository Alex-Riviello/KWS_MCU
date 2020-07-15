# KWS_MCU

## Description

Keyword Spotting on a STM32F446RE microcontroller using the CMSIS-NN library. Simple project to test out preprocessing and neural networks on a microcontroller.   

The model used was TC-ResNet8 from https://arxiv.org/pdf/1904.03814.pdf (with some modifications). It was trained using PyTorch without any residuals on the Google Speech Commands.  

## Details

The project was built using Keil. It requires a license to sucessfully compile.  
The c source code can be found in ./KWS_MCU/Inc and ./KWS_MCU/Src  

Other useful directories:  
./Scripts/TrainedNetwork -> PyTorch model  
./Scripts/Preprocessing -> Code to generate Mel_filters  
./Scripts/ParameterExtraction -> Code to Extract weights from PyTorch model to fixed-point representations   



