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

void binConv2D(const uint32_t* i_act, 
               const uint16_t dim_x,
               const uint16_t dim_y,
               const uint16_t dim_in_ch,
               const uint16_t dim_out_ch,
               const uint16_t dil_x,
               const uint16_t dil_y, 
               const uint16_t pad_x,
               const uint16_t pad_y,
               uint32_t* o_act,
               const uint32_t* wt
               )
{
  int16_t x = 0;
  int16_t y = 0;
  int16_t cur_x = 0;
  int16_t cur_y = 0;
  int16_t bufferA[32] = {0};
  int16_t bufferB[32] = {0};

  /* Iterate over Mel-Frequency Coefficients */
  for(y=-pad_y; y<dim_y-pad_y; y++)
  {
    /* Iterate over time. */
    for(x=-pad_x; x<(dim_x-pad_x); x++)
    {
      /* TOP OF KERNEL */
      cur_x = x;
      cur_y = y;
      if(cur_y >= 0)
      {
        // Case 0 (Top left corner)
        if(cur_x >= 0)
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferA);
        }

        // Case 1 (Top middle)
        cur_x = cur_x + dil_x;
        if((cur_x >= 0) && (cur_x < dim_x))
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }

        // Case 2 (Top right corner)
        cur_x = cur_x + dil_x;
        if(cur_x < dim_x)
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }
      }

      /* MIDDLE OF KERNEL */
      cur_x = x;
      cur_y = cur_y + dil_y;
      if((cur_y >= 0) && (cur_y < dim_y))
      {
        // Case 3 (Middle left)
        if(cur_x >= 0)
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }

        // Case 4 (Middle middle)
        cur_x = cur_x + dil_x;
        if((cur_x >= 0) && (cur_x < dim_x))
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }

        // Case 5 (Middle right)
        cur_x = cur_x + dil_x;
        if(cur_x < dim_x)
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }   
      }

      /* BOTTOM OF KERNEL */
      cur_x = x;
      cur_y = cur_y + dil_y;
      if((cur_y >= 0) && (cur_y < dim_y))
      {
        // Case 3 (Middle left)
        if(cur_x >= 0)
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }

        // Case 4 (Middle middle)
        cur_x = cur_x + dil_x;
        if((cur_x >= 0) && (cur_x < dim_x))
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }

        // Case 5 (Middle right)
        cur_x = cur_x + dil_x;
        if(cur_x < dim_x)
        {
          dotProduct(i_act, cur_x, cur_y, dim_in_ch, dim_x, dim_y, wt, bufferB);
          bufferA[0] += bufferB[0]; // Add other elements (this is temporary)
        }   
      }

    }
  }
  o_act[0] = bufferA[0];
}

void dotProduct(const uint32_t* i_act,
                const uint16_t current_x,
                const uint16_t current_y,
                const uint16_t dim_in_ch,
								const uint16_t dim_x,
                const uint16_t dim_y,
                const uint32_t* wt,
                int16_t* o_temp
                )
{
  uint32_t value = i_act[(dim_x*current_y + current_x)<<2]; // For 32 input channels
  
  uint32_t LUT_addr = (value&0x04)<<5; 
  union lut_v v0[8] = {0};
  union lut_v v1[8] = {0};
  union lut_v v2[8] = {0};
  union lut_v v3[8] = {0};
  union lut_v v4[8] = {0};
  union lut_v v5[8] = {0};
  union lut_v v6[8] = {0};
  union lut_v v7[8] = {0};

  // Bits 0 to 3
  LUT_addr = ((value)&0x04)<<5; 
  v0[0].WORD = wt[LUT_addr++];
  v1[0].WORD = wt[LUT_addr++];
  v2[0].WORD = wt[LUT_addr++];
  v3[0].WORD = wt[LUT_addr++];
  v4[0].WORD = wt[LUT_addr++];
  v5[0].WORD = wt[LUT_addr++];
  v6[0].WORD = wt[LUT_addr++];
  v7[0].WORD = wt[LUT_addr++];

  // Bits 4 to 7
  LUT_addr = ((value>>4)&0x04)<<5; 
  v0[1].WORD = wt[LUT_addr++];
  v1[1].WORD = wt[LUT_addr++];
  v2[1].WORD = wt[LUT_addr++];
  v3[1].WORD = wt[LUT_addr++];
  v4[1].WORD = wt[LUT_addr++];
  v5[1].WORD = wt[LUT_addr++];
  v6[1].WORD = wt[LUT_addr++];
  v7[1].WORD = wt[LUT_addr++];

  // Bits 8 to 11
  LUT_addr = ((value>>8)&0x04)<<5; 
  v0[2].WORD = wt[LUT_addr++];
  v1[2].WORD = wt[LUT_addr++];
  v2[2].WORD = wt[LUT_addr++];
  v3[2].WORD = wt[LUT_addr++];
  v4[2].WORD = wt[LUT_addr++];
  v5[2].WORD = wt[LUT_addr++];
  v6[2].WORD = wt[LUT_addr++];
  v7[2].WORD = wt[LUT_addr++];

  // Bits 12 to 15
  LUT_addr = ((value>>12)&0x04)<<5; 
  v0[3].WORD = wt[LUT_addr++];
  v1[3].WORD = wt[LUT_addr++];
  v2[3].WORD = wt[LUT_addr++];
  v3[3].WORD = wt[LUT_addr++];
  v4[3].WORD = wt[LUT_addr++];
  v5[3].WORD = wt[LUT_addr++];
  v6[3].WORD = wt[LUT_addr++];
  v7[3].WORD = wt[LUT_addr++];

  // Bits 16 to 19
  LUT_addr = ((value>>16)&0x04)<<5; 
  v0[4].WORD = wt[LUT_addr++];
  v1[4].WORD = wt[LUT_addr++];
  v2[4].WORD = wt[LUT_addr++];
  v3[4].WORD = wt[LUT_addr++];
  v4[4].WORD = wt[LUT_addr++];
  v5[4].WORD = wt[LUT_addr++];
  v6[4].WORD = wt[LUT_addr++];
  v7[4].WORD = wt[LUT_addr++];

  // Bits 20 to 23
  LUT_addr = ((value>>20)&0x04)<<5; 
  v0[5].WORD = wt[LUT_addr++];
  v1[5].WORD = wt[LUT_addr++];
  v2[5].WORD = wt[LUT_addr++];
  v3[5].WORD = wt[LUT_addr++];
  v4[5].WORD = wt[LUT_addr++];
  v5[5].WORD = wt[LUT_addr++];
  v6[5].WORD = wt[LUT_addr++];
  v7[5].WORD = wt[LUT_addr++];

  // Bits 24 to 27
  LUT_addr = ((value>>24)&0x04)<<5; 
  v0[6].WORD = wt[LUT_addr++];
  v1[6].WORD = wt[LUT_addr++];
  v2[6].WORD = wt[LUT_addr++];
  v3[6].WORD = wt[LUT_addr++];
  v4[6].WORD = wt[LUT_addr++];
  v5[6].WORD = wt[LUT_addr++];
  v6[6].WORD = wt[LUT_addr++];
  v7[6].WORD = wt[LUT_addr++];

  // Bits 28 to 31
  LUT_addr = ((value>>28)&0x04)<<5; 
  v0[7].WORD = wt[LUT_addr++];
  v1[7].WORD = wt[LUT_addr++];
  v2[7].WORD = wt[LUT_addr++];
  v3[7].WORD = wt[LUT_addr++];
  v4[7].WORD = wt[LUT_addr++];
  v5[7].WORD = wt[LUT_addr++];
  v6[7].WORD = wt[LUT_addr++];
  v7[7].WORD = wt[LUT_addr++];

  // Channels 0-3
  v0[0].WORD = __sadd8(v0[0].WORD, v0[1].WORD);
  v0[0].WORD = __sadd8(v0[0].WORD, v0[2].WORD);
  v0[0].WORD = __sadd8(v0[0].WORD, v0[3].WORD);
  v0[0].WORD = __sadd8(v0[0].WORD, v0[4].WORD);
  v0[0].WORD = __sadd8(v0[0].WORD, v0[5].WORD);
  v0[0].WORD = __sadd8(v0[0].WORD, v0[6].WORD);
  v0[0].WORD = __sadd8(v0[0].WORD, v0[7].WORD);

  o_temp[0] = (int16_t)v0[0].BYTE[0];
  o_temp[1] = (int16_t)v0[0].BYTE[1];
  o_temp[2] = (int16_t)v0[0].BYTE[2];
  o_temp[3] = (int16_t)v0[0].BYTE[3];

  // Channels 4-7
  v1[0].WORD = __sadd8(v1[0].WORD, v1[1].WORD);
  v1[0].WORD = __sadd8(v1[0].WORD, v1[2].WORD);
  v1[0].WORD = __sadd8(v1[0].WORD, v1[3].WORD);
  v1[0].WORD = __sadd8(v1[0].WORD, v1[4].WORD);
  v1[0].WORD = __sadd8(v1[0].WORD, v1[5].WORD);
  v1[0].WORD = __sadd8(v1[0].WORD, v1[6].WORD);
  v1[0].WORD = __sadd8(v1[0].WORD, v1[7].WORD);

  o_temp[4] = (int16_t)v1[0].BYTE[0];
  o_temp[5] = (int16_t)v1[0].BYTE[1];
  o_temp[6] = (int16_t)v1[0].BYTE[2];
  o_temp[7] = (int16_t)v1[0].BYTE[3];

  // Channels 8-11
  v2[0].WORD = __sadd8(v2[0].WORD, v2[1].WORD);
  v2[0].WORD = __sadd8(v2[0].WORD, v2[2].WORD);
  v2[0].WORD = __sadd8(v2[0].WORD, v2[3].WORD);
  v2[0].WORD = __sadd8(v2[0].WORD, v2[4].WORD);
  v2[0].WORD = __sadd8(v2[0].WORD, v2[5].WORD);
  v2[0].WORD = __sadd8(v2[0].WORD, v2[6].WORD);
  v2[0].WORD = __sadd8(v2[0].WORD, v2[7].WORD);

  o_temp[8] = (int16_t)v2[0].BYTE[0];
  o_temp[9] = (int16_t)v2[0].BYTE[1];
  o_temp[10] = (int16_t)v2[0].BYTE[2];
  o_temp[11] = (int16_t)v2[0].BYTE[3];

  // Channels 12-15
  v3[0].WORD = __sadd8(v3[0].WORD, v3[1].WORD);
  v3[0].WORD = __sadd8(v3[0].WORD, v3[2].WORD);
  v3[0].WORD = __sadd8(v3[0].WORD, v3[3].WORD);
  v3[0].WORD = __sadd8(v3[0].WORD, v3[4].WORD);
  v3[0].WORD = __sadd8(v3[0].WORD, v3[5].WORD);
  v3[0].WORD = __sadd8(v3[0].WORD, v3[6].WORD);
  v3[0].WORD = __sadd8(v3[0].WORD, v3[7].WORD);

  o_temp[12] = (int16_t)v3[0].BYTE[0];
  o_temp[13] = (int16_t)v3[0].BYTE[1];
  o_temp[14] = (int16_t)v3[0].BYTE[2];
  o_temp[15] = (int16_t)v3[0].BYTE[3];

  // Channels 16-19
  v4[0].WORD = __sadd8(v4[0].WORD, v4[1].WORD);
  v4[0].WORD = __sadd8(v4[0].WORD, v4[2].WORD);
  v4[0].WORD = __sadd8(v4[0].WORD, v4[3].WORD);
  v4[0].WORD = __sadd8(v4[0].WORD, v4[4].WORD);
  v4[0].WORD = __sadd8(v4[0].WORD, v4[5].WORD);
  v4[0].WORD = __sadd8(v4[0].WORD, v4[6].WORD);
  v4[0].WORD = __sadd8(v4[0].WORD, v4[7].WORD); 

  o_temp[16] = (int16_t)v4[0].BYTE[0];
  o_temp[17] = (int16_t)v4[0].BYTE[1];
  o_temp[18] = (int16_t)v4[0].BYTE[2];
  o_temp[19] = (int16_t)v4[0].BYTE[3];

  // Channels 20-23
  v5[0].WORD = __sadd8(v5[0].WORD, v5[1].WORD);
  v5[0].WORD = __sadd8(v5[0].WORD, v5[2].WORD);
  v5[0].WORD = __sadd8(v5[0].WORD, v5[3].WORD);
  v5[0].WORD = __sadd8(v5[0].WORD, v5[4].WORD);
  v5[0].WORD = __sadd8(v5[0].WORD, v5[5].WORD);
  v5[0].WORD = __sadd8(v5[0].WORD, v5[6].WORD);
  v5[0].WORD = __sadd8(v5[0].WORD, v5[7].WORD);

  o_temp[20] = (int16_t)v5[0].BYTE[0];
  o_temp[21] = (int16_t)v5[0].BYTE[1];
  o_temp[22] = (int16_t)v5[0].BYTE[2];
  o_temp[23] = (int16_t)v5[0].BYTE[3];

  // Channels 24-27
  v6[0].WORD = __sadd8(v6[0].WORD, v6[1].WORD);
  v6[0].WORD = __sadd8(v6[0].WORD, v6[2].WORD);
  v6[0].WORD = __sadd8(v6[0].WORD, v6[3].WORD);
  v6[0].WORD = __sadd8(v6[0].WORD, v6[4].WORD);
  v6[0].WORD = __sadd8(v6[0].WORD, v6[5].WORD);
  v6[0].WORD = __sadd8(v6[0].WORD, v6[6].WORD);
  v6[0].WORD = __sadd8(v6[0].WORD, v6[7].WORD);

  o_temp[24] = (int16_t)v6[0].BYTE[0];
  o_temp[25] = (int16_t)v6[0].BYTE[1];
  o_temp[26] = (int16_t)v6[0].BYTE[2];
  o_temp[27] = (int16_t)v6[0].BYTE[3];

  // Channels 28-31
  v7[0].WORD = __sadd8(v7[0].WORD, v7[1].WORD);
  v7[0].WORD = __sadd8(v7[0].WORD, v7[2].WORD);
  v7[0].WORD = __sadd8(v7[0].WORD, v7[3].WORD);
  v7[0].WORD = __sadd8(v7[0].WORD, v7[4].WORD);
  v7[0].WORD = __sadd8(v7[0].WORD, v7[5].WORD);
  v7[0].WORD = __sadd8(v7[0].WORD, v7[6].WORD);
  v7[0].WORD = __sadd8(v7[0].WORD, v7[7].WORD);

  o_temp[28] = (int16_t)v7[0].BYTE[0];
  o_temp[29] = (int16_t)v7[0].BYTE[1];
  o_temp[30] = (int16_t)v7[0].BYTE[2];
  o_temp[31] = (int16_t)v7[0].BYTE[3];

}

/* -------------------- PARAMETER DECLARATIONS -------------------- */

// Conv0 (3 -> 32)
const int8_t w_conv0[CONV0_IN_CH*CONV0_W_HEIGHT*CONV0_W_WIDTH*LUT_SIZE*CONV0_OUT_CH];

// Conv1 (32 -> 16)
const int8_t w_conv1[CONV1_IN_CH*CONV1_W_HEIGHT*CONV1_W_WIDTH*LUT_SIZE*CONV1_OUT_CH];
