/**
  ******************************************************************************
  * File Name          : bnn_funct.h
  * Description        : This file contains all the functions prototypes for 
  *                      the gpio  
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/

#ifndef __bnn_funct_H
#define __bnn_funct_H

typedef unsigned char   uint8_t;
typedef unsigned short  uint16_t;
typedef unsigned        uint32_t;

typedef signed char     int8_t;
typedef signed short    int16_t;
typedef signed          int32_t;

union lut_v {
  int32_t WORD;
  int8_t BYTE[4];
};

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
               );

void dotProduct(const uint32_t* i_act,
                const uint16_t current_x,
                const uint16_t current_y,
                const uint16_t dim_in_ch,
								const uint16_t dim_x,
                const uint16_t dim_y,
                const uint32_t* wt,
                int16_t* o_temp
                );














#endif

