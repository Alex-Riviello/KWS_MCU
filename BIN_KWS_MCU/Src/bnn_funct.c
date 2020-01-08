/**
  ******************************************************************************
  * @file           : bnn_funct.c
  * @brief          : 
  ******************************************************************************
  * @attention
  *
  *
  ******************************************************************************
  */

#include "bnn_funct.h"

#define LUT_IN 4
#define LUT_SIZE 16

#define CONV0_IN_CH 3
#define CONV0_W_HEIGHT 3
#define CONV0_W_WIDTH 3
#define CONV0_OUT_CH 32

#define CONV1_IN_CH 32/LUT_IN
#define CONV1_W_HEIGHT 3
#define CONV1_W_WIDTH 3
#define CONV1_OUT_CH 16



/* -------------------- PARAMETER DECLARATIONS -------------------- */

// Conv0 (3 -> 32)
const int8_t w_conv0[CONV0_IN_CH*CONV0_W_HEIGHT*CONV0_W_WIDTH*LUT_SIZE*CONV0_OUT_CH];

// Conv1 (32 -> 16)
const int8_t w_conv1[CONV1_IN_CH*CONV1_W_HEIGHT*CONV1_W_WIDTH*LUT_SIZE*CONV1_OUT_CH];
