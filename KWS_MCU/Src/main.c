
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

/* USER CODE BEGIN Includes */
#define ARM_MATH_CM4
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
float current_frame[N_FFT];
float FFT_in[N_FFT];
float FFT_out[N_FFT];
float MAG_out[N_FFT/2];
volatile int adc_done = 0;
int i = 0;
int j = 0;
int k = 0;
int frame_pos = 0;


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);

/* USER CODE BEGIN PFP */
/* Private function prototypes -----------------------------------------------*/
uint16_t pixel_color(float coeff_value);
void compute_logMelCoefficients(int frame_position);

/* USER CODE END PFP */

/* USER CODE BEGIN 0 */
uint16_t pixel_color(float coeff_value)
{
	if (coeff_value > 60)
		return MAGENTA;
	else if (coeff_value > 45)
		return RED;
	else if (coeff_value > 30)
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
	
	if (frame_position<2)
		return;
	
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
		
	// Apply Filterbank
	for(j=0; j<N_MEL; j++)
	{
		spectrogram[frame_position*N_MEL + j] = 0.0;
		for(k=0; k<32; k++)
		{
			int indice = FILTER_INDICES[j];
			spectrogram[frame_position*N_MEL + j] = spectrogram[frame_position*N_MEL + j] + MAG_out[indice+k]*FILTERBANK[32*j+k];
		}
		spectrogram[frame_position*N_MEL + j] = (float)50*(log10(spectrogram[frame_position*N_MEL + j])-3.0);
		uint16_t color = pixel_color(spectrogram[frame_position*N_MEL + j]);
		LCD_DrawPixel(j+160, frame_position+70, color);
	}
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
