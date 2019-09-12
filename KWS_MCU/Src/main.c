
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  ** This notice applies to any and all portions of this file
  * that are not between comment pairs USER CODE BEGIN and
  * USER CODE END. Other portions of this file, whether 
  * inserted by the user or by software development tools
  * are owned by their respective copyright owners.
  *
  * COPYRIGHT(c) 2019 STMicroelectronics
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32f4xx_hal.h"
#include "adc.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"
#include "filter_values.h"
#include "network_weights.h"
#include "network_parameters.h"
#include "arm_nnfunctions.h"

/* USER CODE BEGIN Includes */

#include "arm_math.h"
#include "core_cm4.h"
#include "arm_const_structs.h"
#include "LCD.h"

/* USER CODE END Includes */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/
#define N_FFT 512
#define N_SHIFT 160
#define N_MEL 40
#define N_FRAMES 100
#define RFFT 0
#define RIFFT 1
	
uint32_t audio_input_A[N_SHIFT];
uint32_t audio_input_B[N_SHIFT];
uint32_t * current_audio_input = audio_input_A;
float spectrogram[N_FRAMES*N_MEL];
q7_t q7_spectrogram[N_FRAMES*N_MEL] = TEST_INPUT;
float current_frame[N_FFT];
float DC_value[1];
float FFT_in[N_FFT];
float FFT_out[N_FFT];
float MAG_out[N_FFT/2];
volatile int adc_done = 0;
int i = 0;
int j = 0;
int k = 0;
int frame_pos = 0;

// include the input and weights
static q7_t conv0_wt[CONV0_IM_CH * CONV0_KER_DIM_X * CONV0_KER_DIM_Y * CONV0_OUT_CH] = CONV0_WT;
static q7_t conv0_bias[CONV0_OUT_CH] = CONV0_BIAS;
static q7_t conv1_wt[CONV1_IM_CH * CONV1_KER_DIM_X * CONV1_KER_DIM_Y * CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;
static q7_t conv2_wt[CONV2_IM_CH * CONV2_KER_DIM_X * CONV2_KER_DIM_Y * CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;
static q7_t conv3_wt[CONV3_IM_CH * CONV3_KER_DIM_X * CONV3_KER_DIM_Y * CONV3_OUT_CH] = CONV3_WT;
static q7_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;
static q7_t conv4_wt[CONV4_IM_CH * CONV4_KER_DIM_X * CONV4_KER_DIM_Y * CONV4_OUT_CH] = CONV4_WT;
static q7_t conv4_bias[CONV4_OUT_CH] = CONV4_BIAS;
static q7_t conv5_wt[CONV5_IM_CH * CONV5_KER_DIM_X * CONV5_KER_DIM_Y * CONV5_OUT_CH] = CONV5_WT;
static q7_t conv5_bias[CONV5_OUT_CH] = CONV5_BIAS;
static q7_t conv6_wt[CONV6_IM_CH * CONV6_KER_DIM_X * CONV6_KER_DIM_Y * CONV6_OUT_CH] = CONV6_WT;
static q7_t conv6_bias[CONV6_OUT_CH] = CONV6_BIAS;

static q7_t ip1_wt[IP1_DIM * IP1_OUT] = IP1_WT;
static q7_t ip1_bias[IP1_OUT] = IP1_BIAS;

q7_t img_buffer1[1600];
q7_t img_buffer2[1600];
q7_t mean_buffer[48*13];
q7_t fc_input[48];
q7_t fc_output[12];
q7_t softmax_output[12];


q15_t bufferA[2*CONV6_IM_CH*CONV1_KER_DIM_X*CONV1_KER_DIM_Y];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);

/* USER CODE BEGIN PFP */
/* Private function prototypes -----------------------------------------------*/
uint16_t pixel_color(float coeff_value);
void draw_square(int16_t x, int16_t y, uint16_t color);
void compute_logMelCoefficients(int frame_position);
uint8_t TC_ResNet(void);

/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

void print_class(q7_t * output_prob)
{
	q7_t val_result[1];
	uint32_t index_result[1];
	arm_max_q7(output_prob, 12, val_result, index_result);
	switch(index_result[0])
	{
		case 0: LCD_Printf("Silence\r\n"); break;
		case 1: LCD_Printf("Unknown\r\n"); break;
		case 2: LCD_Printf("Yes\r\n"); break;
		case 3: LCD_Printf("No\r\n"); break;
		case 4: LCD_Printf("Up\r\n"); break;
		case 5: LCD_Printf("Down\r\n"); break;
		case 6: LCD_Printf("Left\r\n"); break;
		case 7: LCD_Printf("Right\r\n"); break;
		case 8: LCD_Printf("On\r\n"); break;
		case 9: LCD_Printf("Off\r\n"); break;
		case 10: LCD_Printf("Stop\r\n"); break;
		case 11: LCD_Printf("Go\r\n"); break;
	}
	
}

void draw_square(int16_t x, int16_t y, uint16_t color)
{
	LCD_DrawPixel(x, y, color);
	LCD_DrawPixel(x+1, y, color);
	LCD_DrawPixel(x, y+1, color);
	LCD_DrawPixel(x+1, y+1, color);
}

uint16_t pixel_color(float coeff_value)
{
	if (coeff_value > 0.5)
		return MAGENTA;
	else if (coeff_value > 0.0)
		return RED;
	else if (coeff_value > -0.5)
		return YELLOW;
	else
		return CYAN;
}

void compute_logMelCoefficients(int frame_position)
{
	//Shifting old values
	for(j=N_SHIFT; j<2*N_SHIFT; j++)
	{
		current_frame[j+N_SHIFT] = current_frame[j];
	}
	for(j=0; j<N_SHIFT; j++)
	{
		current_frame[j+N_SHIFT] = current_frame[j];
	}
	
	//Conversion from uint32 to float
	for(j=0; j<N_SHIFT; j++)
	{
		current_frame[j] = audio_input_A[j]; 
	}
	// Substract average to remove DC
	arm_mean_f32(current_frame, N_SHIFT, DC_value);
	for(j=0; j<N_SHIFT; j++)
	{
		current_frame[j] = current_frame[j] - DC_value[0];
	}
	
	//if (frame_position<2)
	//	return;
	
	for(j=0; j<480; j++)
	{
		FFT_in[j] = current_frame[j]*HAMMING_WINDOW[j];
	}
	for(j=480; j<N_FFT; j++)
	{
		FFT_in[j] = 0.0;
	}
	// Compute one FFT and its magnitude
			
	arm_rfft_fast_instance_f32 fftInstance;
	arm_rfft_fast_init_f32(&fftInstance, N_FFT);
	arm_rfft_fast_f32(&fftInstance, FFT_in, FFT_out, 0);
	arm_cmplx_mag_f32(FFT_out, MAG_out, (N_FFT/2));
	
	// Clear the buffer
	for(j=0; j<N_MEL; j++)
	{
		spectrogram[frame_position*N_MEL + j] = 0.0;
	}
	
	// Apply Filterbank
	for(j=0; j<N_MEL; j++)
	{
		for(k=0; k<32; k++)
		{
			int indice = FILTER_INDICES[j];
			spectrogram[frame_position*N_MEL + j] = spectrogram[frame_position*N_MEL + j] + MAG_out[indice+k]*FILTERBANK[32*j+k];
		}
		//spectrogram[frame_position*N_MEL + j] = (float)40.0*log(spectrogram[frame_position*N_MEL + j])-380.0;
		spectrogram[frame_position*N_MEL + j] = (float)0.2*log(spectrogram[frame_position*N_MEL + j])-1.6;
		uint16_t color = pixel_color(spectrogram[frame_position*N_MEL + j]);
		//LCD_DrawPixel(j+160, frame_position+70, color);
		draw_square(140+2*j, 20 + 2*frame_position, color);
	}
}

uint8_t TC_ResNet()
{
	// Conv0
  arm_convolve_HWC_q7_basic_nonsquare(q7_spectrogram, CONV0_IM_DIM_X, CONV0_IM_DIM_Y, CONV0_IM_CH, conv0_wt, CONV0_OUT_CH, CONV0_KER_DIM_X,
																		CONV0_KER_DIM_Y, CONV0_PADDING_X, CONV0_PADDING_Y, CONV0_STRIDE_X, CONV0_STRIDE_Y, conv0_bias,
																		CONV0_BIAS_LSHIFT, CONV0_OUT_RSHIFT, img_buffer1, CONV0_OUT_DIM_X, CONV0_OUT_DIM_Y, bufferA, NULL);
	arm_relu_q7(img_buffer1, CONV0_OUT_DIM_X*CONV0_OUT_DIM_Y*CONV0_OUT_CH);
	
	// Conv1
	arm_convolve_HWC_q7_basic_nonsquare(img_buffer1, CONV1_IM_DIM_X, CONV1_IM_DIM_Y, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM_X,
																		CONV1_KER_DIM_Y, CONV1_PADDING_X, CONV1_PADDING_Y, CONV1_STRIDE_X, CONV1_STRIDE_Y, conv1_bias,
																		CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, img_buffer2, CONV1_OUT_DIM_X, CONV1_OUT_DIM_Y, bufferA, NULL);
	arm_relu_q7(img_buffer2, CONV1_OUT_DIM_X*CONV1_OUT_DIM_Y*CONV1_OUT_CH);
	
	// Conv2
	arm_convolve_HWC_q7_basic_nonsquare(img_buffer2, CONV2_IM_DIM_X, CONV2_IM_DIM_Y, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM_X,
																		CONV2_KER_DIM_Y, CONV2_PADDING_X, CONV2_PADDING_Y, CONV2_STRIDE_X, CONV2_STRIDE_Y, conv2_bias,
																		CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, img_buffer1, CONV2_OUT_DIM_X, CONV2_OUT_DIM_Y, bufferA, NULL);
	arm_relu_q7(img_buffer1, CONV2_OUT_DIM_X*CONV2_OUT_DIM_Y*CONV2_OUT_CH);
	
	// Conv3
	arm_convolve_HWC_q7_basic_nonsquare(img_buffer1, CONV3_IM_DIM_X, CONV3_IM_DIM_Y, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM_X,
																		CONV3_KER_DIM_Y, CONV3_PADDING_X, CONV3_PADDING_Y, CONV3_STRIDE_X, CONV3_STRIDE_Y, conv3_bias,
																		CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, img_buffer2, CONV3_OUT_DIM_X, CONV3_OUT_DIM_Y, bufferA, NULL);
	arm_relu_q7(img_buffer2, CONV3_OUT_DIM_X*CONV3_OUT_DIM_Y*CONV3_OUT_CH);
	
	// Conv4
	arm_convolve_HWC_q7_basic_nonsquare(img_buffer2, CONV4_IM_DIM_X, CONV4_IM_DIM_Y, CONV4_IM_CH, conv4_wt, CONV4_OUT_CH, CONV4_KER_DIM_X,
																		CONV4_KER_DIM_Y, CONV4_PADDING_X, CONV4_PADDING_Y, CONV4_STRIDE_X, CONV4_STRIDE_Y, conv4_bias,
																		CONV4_BIAS_LSHIFT, CONV4_OUT_RSHIFT, img_buffer1, CONV4_OUT_DIM_X, CONV4_OUT_DIM_Y, bufferA, NULL);
	arm_relu_q7(img_buffer1, CONV4_OUT_DIM_X*CONV4_OUT_DIM_Y*CONV4_OUT_CH);
	
	// Conv5
	arm_convolve_HWC_q7_basic_nonsquare(img_buffer1, CONV5_IM_DIM_X, CONV5_IM_DIM_Y, CONV5_IM_CH, conv5_wt, CONV5_OUT_CH, CONV5_KER_DIM_X,
																		CONV5_KER_DIM_Y, CONV5_PADDING_X, CONV5_PADDING_Y, CONV5_STRIDE_X, CONV5_STRIDE_Y, conv5_bias,
																		CONV5_BIAS_LSHIFT, CONV5_OUT_RSHIFT, img_buffer2, CONV5_OUT_DIM_X, CONV5_OUT_DIM_Y, bufferA, NULL);
	arm_relu_q7(img_buffer2, CONV5_OUT_DIM_X*CONV5_OUT_DIM_Y*CONV5_OUT_CH);
	
	// Conv6
	arm_convolve_HWC_q7_basic_nonsquare(img_buffer2, CONV6_IM_DIM_X, CONV6_IM_DIM_Y, CONV6_IM_CH, conv6_wt, CONV6_OUT_CH, CONV6_KER_DIM_X,
																		CONV6_KER_DIM_Y, CONV6_PADDING_X, CONV6_PADDING_Y, CONV6_STRIDE_X, CONV6_STRIDE_Y, conv6_bias,
																		CONV6_BIAS_LSHIFT, CONV6_OUT_RSHIFT, img_buffer1, CONV6_OUT_DIM_X, CONV6_OUT_DIM_Y, bufferA, NULL);
	arm_relu_q7(img_buffer1, CONV6_OUT_DIM_X*CONV6_OUT_DIM_Y*CONV6_OUT_CH);
	
	for(i=0; i<CONV6_OUT_CH; i++){
		for(j=0; j<CONV6_OUT_DIM_Y; j++){
			mean_buffer[CONV6_OUT_DIM_Y*i+j] = img_buffer1[CONV6_OUT_CH*j + i];
		}
	}
	
	for(i=0; i<CONV6_OUT_CH; i++){
		arm_mean_q7(&mean_buffer[CONV6_OUT_DIM_Y*i], CONV6_OUT_DIM_Y ,&fc_input[i]);
	}
	
	// Everything upto here works (about same results as Python for an array of 1s as input)
	arm_fully_connected_q7(fc_input, ip1_wt, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, fc_output, bufferA);
	
	arm_softmax_q7(fc_output, IP1_OUT, softmax_output);
	
	
	return 0;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  *
  * @retval None
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	
  /* USER CODE END 1 */

  /* MCU Configuration----------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART2_UART_Init();
  MX_ADC1_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */
	LCD_Begin();
	HAL_Delay(20);
	LCD_SetRotation(1);
	LCD_FillScreen(BLACK);
	//LCD_Printf("! Hello !");
	
	// Enabling ADC and DMA timer
	HAL_TIM_Base_Start(&htim2);
	HAL_ADC_Start_DMA(&hadc1, current_audio_input, N_SHIFT);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {

  /* USER CODE END WHILE */

  /* USER CODE BEGIN 3 */
		while (adc_done == 0) {}; 
		adc_done = 0;
		
		//Compute and print spectrogram
		compute_logMelCoefficients(frame_pos);
		
		
		//Updating frame position
		if(frame_pos > 100)
		{
			frame_pos = 0;
			LCD_FillScreen(BLACK);
			LCD_SetRotation(2);
			LCD_SetCursor(0, 0);
			arm_float_to_q7(spectrogram, q7_spectrogram, N_MEL*N_FRAMES);
			TC_ResNet();
			print_class(softmax_output);
			LCD_SetRotation(1);
		}
		else
		{
			frame_pos = frame_pos + 1;
		}
		
  }
  /* USER CODE END 3 */

}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{

  RCC_OscInitTypeDef RCC_OscInitStruct;
  RCC_ClkInitTypeDef RCC_ClkInitStruct;

    /**Configure the main internal regulator output voltage 
    */
  __HAL_RCC_PWR_CLK_ENABLE();

  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

    /**Initializes the CPU, AHB and APB busses clocks 
    */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = 16;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    _Error_Handler(__FILE__, __LINE__);
  }

    /**Initializes the CPU, AHB and APB busses clocks 
    */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    _Error_Handler(__FILE__, __LINE__);
  }

    /**Configure the Systick interrupt time 
    */
  HAL_SYSTICK_Config(HAL_RCC_GetHCLKFreq()/1000);

    /**Configure the Systick 
    */
  HAL_SYSTICK_CLKSourceConfig(SYSTICK_CLKSOURCE_HCLK);

  /* SysTick_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(SysTick_IRQn, 0, 0);
}

/* USER CODE BEGIN 4 */

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc) {
	if (hadc->Instance == ADC1) {
		//switch buffer
		if (current_audio_input == audio_input_A) 
			current_audio_input = audio_input_B;
		else
			current_audio_input = audio_input_A;
		adc_done = 1;
	}
}


/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  file: The file name as string.
  * @param  line: The line in file as a number.
  * @retval None
  */
void _Error_Handler(char *file, int line)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  while(1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t* file, uint32_t line)
{ 
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
