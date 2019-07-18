#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include "LCD.h"
#include "stm32f4xx_hal.h"
#include "glcdfont.h"

#define TFTWIDTH   240
#define TFTHEIGHT  320

#define TFTLCD_DELAY 0xFF

// GPIO to data bus pin connections
// ---- PORT Pin ---     --- Data ----
// GPIOA, GPIO_PIN_9  -> BIT 0
// GPIOC, GPIO_PIN_7  -> BIT 1
// GPIOA, GPIO_PIN_10 -> BIT 2
// GPIOB, GPIO_PIN_3  -> BIT 3
// GPIOB, GPIO_PIN_5  -> BIT 4
// GPIOB, GPIO_PIN_4  -> BIT 5
// GPIOB, GPIO_PIN_10 -> BIT 6
// GPIOA, GPIO_PIN_8  -> BIT 7

#define LCD_CS_PIN  GPIO_PIN_0        // PB0 -> A3 // Chip Select goes to Analog 3
#define LCD_CD_PIN  GPIO_PIN_4        // PA4 -> A2 // Command/Data goes to Analog 2
#define LCD_WR_PIN  GPIO_PIN_1        // PA1 -> A1 // LCD Write goes to Analog 1
#define LCD_RD_PIN  GPIO_PIN_0        // PA0 -> A0 // LCD Read goes to Analog 0
#define LCD_RST_PIN GPIO_PIN_1        // PC1 -> RESET


#define LCD_CS_GPIO_PORT  GPIOB
#define LCD_CS_HIGH()     HAL_GPIO_WritePin(LCD_CS_GPIO_PORT, LCD_CS_PIN, GPIO_PIN_SET)
#define LCD_CS_LOW()      HAL_GPIO_WritePin(LCD_CS_GPIO_PORT, LCD_CS_PIN, GPIO_PIN_RESET)

#define LCD_RD_GPIO_PORT  GPIOA
#define LCD_RD_HIGH()     HAL_GPIO_WritePin(LCD_RD_GPIO_PORT, LCD_RD_PIN, GPIO_PIN_SET)
#define LCD_RD_LOW()      HAL_GPIO_WritePin(LCD_RD_GPIO_PORT, LCD_RD_PIN, GPIO_PIN_RESET)

#define LCD_WR_GPIO_PORT  GPIOA
#define LCD_WR_HIGH()                  HAL_GPIO_WritePin(LCD_WR_GPIO_PORT, LCD_WR_PIN, GPIO_PIN_SET)
#define LCD_WR_LOW()      HAL_GPIO_WritePin(LCD_WR_GPIO_PORT, LCD_WR_PIN, GPIO_PIN_RESET)

#define LCD_CD_GPIO_PORT  GPIOA
#define LCD_CD_HIGH()     HAL_GPIO_WritePin(LCD_CD_GPIO_PORT, LCD_CD_PIN, GPIO_PIN_SET)
#define LCD_CD_LOW()      HAL_GPIO_WritePin(LCD_CD_GPIO_PORT, LCD_CD_PIN, GPIO_PIN_RESET)

#define LCD_RST_GPIO_PORT GPIOC
#define LCD_RST_HIGH()    HAL_GPIO_WritePin(LCD_RST_GPIO_PORT, LCD_RST_PIN, GPIO_PIN_SET)
#define LCD_RST_LOW()     HAL_GPIO_WritePin(LCD_RST_GPIO_PORT, LCD_RST_PIN, GPIO_PIN_RESET)


#define LCD_WR_STROBE() { LCD_WR_LOW(); delay(1); LCD_WR_HIGH(); delay(1); }

#define swap(a, b) { int16_t t = a; a = b; b = t; }

static int16_t m_width;
static int16_t m_height;
static int16_t m_cursor_x;
static int16_t m_cursor_y;

static uint16_t m_textcolor;
static uint16_t m_textbgcolor;
static uint8_t m_textsize;
static uint8_t m_rotation;
static uint8_t m_wrap;

static void LCD_Register_Init(void);

// Initialization commands
//
#ifdef SUPPORT_ECRAN_1

static const uint16_t ST7781_regValues[] = {

      0x0000, 0x0000,
            0x0000, 0x0000,
            0x0000, 0x0000,
            0x0000, 0x0001,
            0x00A4, 0x0001,     //CALB=1
            TFTLCD_DELAY, 10,
            0x0060, 0x2700,     //NL
            0x0008, 0x0808,     //FP & BP
            0x0030, 0x0214,     //Gamma settings
            0x0031, 0x3715,
            0x0032, 0x0604,
            0x0033, 0x0E16,
            0x0034, 0x2211,
            0x0035, 0x1500,
            0x0036, 0x8507,
            0x0037, 0x1407,
            0x0038, 0x1403,
            0x0039, 0x0020,
            0x0090, 0x0015,     //DIVI & RTNI
            0x0010, 0x0410,     //BT=4,AP=1
            0x0011, 0x0237,     //DC1=2,DC0=3, VC=7
            0x0029, 0x0046,     //VCM1=70
            0x002A, 0x0046,     //VCMSEL=0,VCM2=70
            // Sleep mode IN sequence
            0x0007, 0x0000,
            //0x0012, 0x0000,   //PSON=0,PON=0
            // Sleep mode EXIT sequence 
            0x0012, 0x0189,     //VCMR=1,PSON=0,PON=0,VRH=9
            0x0013, 0x1100,     //VDV=17
            TFTLCD_DELAY, 150,
            0x0012, 0x01B9,     //VCMR=1,PSON=1,PON=1,VRH=9 [018F]
//            0x0001, 0x0100,     //SS=1 Other mode settings
            0x0002, 0x0200,     //BC0=1--Line inversion
            0x0003, 0x1030,
            0x0009, 0x0001,     //ISC=1 [0000]
            0x000A, 0x0000,     // [0000]
            //            0x000C, 0x0001,   //RIM=1 [0000]
            0x000D, 0x0000,     // [0000]
            0x000E, 0x0030,     //VEM=3 VCOM equalize [0000]
            0x0061, 0x0001,
            0x006A, 0x0000,
            0x0080, 0x0000,
            0x0081, 0x0000,
            0x0082, 0x005F,
            0x0092, 0x0100,
            0x0093, 0x0701,
            TFTLCD_DELAY, 80,
            0x0007, 0x0100,     //BASEE=1--Display On
        };
                                                           
#elif defined(SUPPORT_ECRAN_2)
static const uint16_t ST7781_regValues[] = {
						0x00FF, 0x0001,     //can we do 0xFF
            0x00F3, 0x0008,
            //  LCD_Write_COM(0x00F3,

            0x00, 0x0001,
            0x0001, 0x0100,     // Driver Output Control Register (R01h)
            0x0002, 0x0700,     // LCD Driving Waveform Control (R02h)
            0x0003, 0x1030,     // Entry Mode (R03h)
            0x0008, 0x0302,
            0x0009, 0x0000,
            0x0010, 0x0000,     // Power Control 1 (R10h)
            0x0011, 0x0007,     // Power Control 2 (R11h)
            0x0012, 0x0000,     // Power Control 3 (R12h)
            0x0013, 0x0000,     // Power Control 4 (R13h)
            TFTLCD_DELAY, 50,
            0x0010, 0x14B0,     // Power Control 1 SAP=1, BT=4, APE=1, AP=3
            TFTLCD_DELAY, 10,
            0x0011, 0x0007,     // Power Control 2 VC=7
            TFTLCD_DELAY, 10,
            0x0012, 0x008E,     // Power Control 3 VCIRE=1, VRH=14
            0x0013, 0x0C00,     // Power Control 4 VDV=12
            0x0029, 0x0015,     // NVM read data 2 VCM=21
            TFTLCD_DELAY, 10,
            0x0030, 0x0000,     // Gamma Control 1
            0x0031, 0x0107,     // Gamma Control 2
            0x0032, 0x0000,     // Gamma Control 3
            0x0035, 0x0203,     // Gamma Control 6
            0x0036, 0x0402,     // Gamma Control 7
            0x0037, 0x0000,     // Gamma Control 8
            0x0038, 0x0207,     // Gamma Control 9
            0x0039, 0x0000,     // Gamma Control 10
            0x003C, 0x0203,     // Gamma Control 13
            0x003D, 0x0403,     // Gamma Control 14
            0x0060, 0xA700,     // Driver Output Control (R60h) .kbv was 0xa700
            0x0061, 0x0001,     // Driver Output Control (R61h)
            0x0090, 0X0029,     // Panel Interface Control 1 (R90h)

            // Display On
            0x0007, 0x0133,     // Display Control (R07h)
            TFTLCD_DELAY, 50,
        };
#elif	defined(SUPPORT_ECRAN_3)
static const uint16_t ST7781_regValues[] = {
			0x00e5,0x8000,
			0x0000,0x0001, //ID Code, here it is 0x0001
			0x0001,0x0100, // shift direction, S720 to S1
			0x0002,0x0700, //line inversion is selected
			0x0003,0x1030, //allows MSB = Red, write from Left to Right
			0x0004,0x0000, //No resize
			0x0008,0x0202, // ?
			0x0009,0x0000, // ?
			0x000a,0x0000, // 1 Frame
			0x000c,0x0000, // ?
			0x000d,0x0000,
			0x000f,0x0000,
			0x0010,0x0000,
			0x0011,0x0000,
			0x0012,0x0000,
			0x0013,0x0000,
			0x0010,0x17b0, //no standbye
			0x0011,0x0037,
			0x0012,0x0138,
			0x0013,0x1700,
			0x0029,0x000d,
			0x0020,0x0000,
			0x0021,0x0000,
			0x0030,0x0001,
			0x0031,0x0606,
			0x0032,0x0304,
			0x0033,0x0202,
			0x0034,0x0202,
			0x0035,0x0103,
			0x0036,0x011d,
			0x0037,0x0404,
			0x0038,0x0404,
			0x0039,0x0404,
			0x003c,0x0700,
			0x003d,0x0a1f,
			0x0050,0x0000,
			0x0051,0x00ef, //TFTWIDTH - 1
			0x0052,0x0000,
			0x0053,0x013f, //TFTHEIGHT - 1
			0x0060,0xA700, //TFTHEIGHT / 8, Also invert image
			0x0061,0x0001,
			0x006a,0x0000,
			0x0090,0x0010, //16 clock per line, Fclk = Fosc
			0x0092,0x0000,
			0x0093,0x0003,
			0x0095,0x0101,
			0x0097,0x0000,
			0x0098,0x0000,
			0x0007,0x0021,
			0x0007,0x0133};
		
#endif        

void delay(unsigned int t)
{
        for (; t > 0; t-- )
        {
        __asm("nop");
        }
}

static void LCD_Register_Init(void)
{	
	uint16_t i,addr,data;
	for(i = 0; i < (sizeof(ST7781_regValues) / sizeof(uint16_t)); i = i+2) {
		addr = ST7781_regValues[i];
		data = ST7781_regValues[i+1];
		if(addr == TFTLCD_DELAY) {
			delay(data);
		} else {
			LCD_WriteRegister16(addr, data);
		}
	}
}


/**
 * \brief GPIO Initialization
 * 
 * \param 
 * 
 * \return void
 */
static void GPIO_Init(void)
{
        GPIO_InitTypeDef GPIO_InitStruct;

        /* GPIO Ports Clock Enable */
        __GPIOC_CLK_ENABLE();
        __GPIOA_CLK_ENABLE();
        __GPIOB_CLK_ENABLE();

        /*Configure GPIO pins : PC1 PC7 */
        GPIO_InitStruct.Pin = GPIO_PIN_1|GPIO_PIN_7;
        GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_LOW;
        HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

        /*Configure GPIO pins : PA0 PA8 PA9 PA10 PA1 PA4*/
        GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_8
                                          |GPIO_PIN_9|GPIO_PIN_10 | GPIO_PIN_1|GPIO_PIN_4;
        GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_LOW;
        HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

        /*Configure GPIO pins : PB0 PB3 PB10 PB4 PB5 */
        GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_3|GPIO_PIN_10|GPIO_PIN_4|GPIO_PIN_5;
        GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_LOW;
        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
}

/**
 * \brief LCD Initialization
 * 
 * \param 
 * 
 * \return void
 */
void LCD_Begin(void)
{
	m_width     = TFTWIDTH;
	m_height    = TFTHEIGHT;
	m_rotation  = 0;
	m_cursor_y  = m_cursor_x    = 0;
	m_textsize  = 4;
	m_textcolor = m_textbgcolor = 0xFFFF;
	m_wrap      = 1;

	GPIO_Init();

	LCD_Reset();

	LCD_CS_LOW();

	LCD_Register_Init();

    LCD_SetRotation(m_rotation);
	LCD_SetAddrWindow(0, 0, TFTWIDTH-1, TFTHEIGHT-1);
}

/**
 * \brief Calucalte 16Bit-RGB
 * 
 * \param r        Red
 * \param g        Green
 * \param b        Blue
 * 
 * \return uint16_t        16Bit-RGB
 */
uint16_t LCD_Color565(uint8_t r, uint8_t g, uint8_t b)
{
  return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

/**
 * \brief Draws a point at the specified coordinates
 * 
 * \param x                x-Coordinate
 * \param y                y-Coordinate
 * \param color        Color
 * 
 * \return void
 */
void LCD_DrawPixel(int16_t x, int16_t y, uint16_t color)
{
        // Clip
        if((x < 0) || (y < 0) || (x >= TFTWIDTH) || (y >= TFTHEIGHT)) return;

        LCD_CS_LOW();

        int16_t t;
        switch(m_rotation) {
        case 1:
                t = x;
                x = TFTWIDTH  - 1 - y;
                y = t;
                break;
        case 2:
                x = TFTWIDTH  - 1 - x;
                y = TFTHEIGHT - 1 - y;
                break;
        case 3:
                t = x;
                x = y;
                y = TFTHEIGHT - 1 - t;
                break;
        }

        LCD_WriteRegister16(0x0020, x);
        LCD_WriteRegister16(0x0021, y);
        LCD_WriteRegister16(0x0022, color);

        LCD_CS_HIGH();
}


/**
 * \brief Draws a line connecting the two points specified by the coordinate pairs
 * 
 * \param x0        The x-coordinate of the first point
 * \param y0        The y-coordinate of the first point
 * \param x1        The x-coordinate of the second point
 * \param y1        The y-coordinate of the second point.
 * \param color        Color
 * 
 * \return void
 */
void LCD_DrawLine(int16_t x1, int16_t y1, int16_t x2, int16_t y2, uint16_t color)
{
        // Bresenham's algorithm - thx wikpedia
        
        int16_t steep = abs(y2 - y1) > abs(x2 - x1);
        if (steep) {
                swap(x1, y1);
                swap(x2, y2);
        }

        if (x1 > x2) {
                swap(x1, x2);
                swap(y1, y2);
        }

        int16_t dx, dy;
        dx = x2 - x1;
        dy = abs(y2 - y1);

        int16_t err = dx / 2;
        int16_t ystep;

        if (y1 < y2) {
                ystep = 1;
        } else {
                ystep = -1;
        }

        for (; x1<=x2; x1++) {
                if (steep) {
                        LCD_DrawPixel(y1, x1, color);
                } else {
                        LCD_DrawPixel(x1, y1, color);
                }
                err -= dy;
                if (err < 0) {
                        y1 += ystep;
                err += dx;
                }
        }
}

/**
 * \brief Draws a horizontal line
 *
 * \param x                        The x-coordinate of the first point
 * \param y                        The y-coordinate of the first point
 * \param length        Length of the line
 * \param color        Color
 * 
 * \return void
 */
void LCD_DrawFastHLine(int16_t x, int16_t y, int16_t length, uint16_t color)
{
        int16_t x2;

        // Initial off-screen clipping
        if((length <= 0) || (y <  0) || (y >= m_height) ||
                        (x >= m_width) || ((x2 = (x+length-1)) <  0))
                return;

        if(x < 0) { // Clip left
                length += x;
                x = 0;
        }

        if(x2 >= m_width) { // Clip right
                x2      = m_width - 1;
                length  = x2 - x + 1;
        }

        LCD_SetAddrWindow(x, y, x2, y);
        LCD_Flood(color, length);
        LCD_SetAddrWindow(0, 0, m_width - 1, m_height - 1);

}

/**
 * \brief Draws a vertical line
 *
 * \param x                The x-coordinate of the first point
 * \param y                The y-coordinate of the first point
 * \param h                High of the line
 * \param color        Color
 * 
 * \return void
 */
void LCD_DrawFastVLine(int16_t x, int16_t y, int16_t h, uint16_t color)
{
        // Update in subclasses if desired!
        LCD_DrawLine(x, y, x, y+h-1, color);
}

/**
 * \brief Draws a rectangle specified by a coordinate pair, a width, and a height.
 * 
 * \param x                        The x-coordinate of the upper-left corner of the rectangle to draw
 * \param y                        The y-coordinate of the upper-left corner of the rectangle to draw
 * \param w                        Width of the rectangle to draw
 * \param h                        Height of the rectangle to draw
 * \param color                Color
 * 
 * \return void
 */
void LCD_DrawRect(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t color)
{
        LCD_DrawFastHLine(x, y, w, color);
        LCD_DrawFastHLine(x, y+h-1, w, color);
        LCD_DrawFastVLine(x, y, h, color);
        LCD_DrawFastVLine(x+w-1, y, h, color);
}


/**
 * \brief Draws a rectangle with rounded corners specified by a coordinate pair, a width, and a height.
 * 
 * \param x                        The x-coordinate of the upper-left corner of the rectangle to draw
 * \param y                        The y-coordinate of the upper-left corner of the rectangle to draw
 * \param w                        Width of the rectangle to draw
 * \param h                        Height of the rectangle to draw
 * \param r                        Radius
 * \param color                Color
 * 
 * \return void
 */
void LCD_DrawRoundRect(int16_t x, int16_t y, int16_t w, int16_t h, int16_t r, uint16_t color)
{
        // smarter version
        LCD_DrawFastHLine(x+r  , y    , w-2*r, color); // Top
        LCD_DrawFastHLine(x+r  , y+h-1, w-2*r, color); // Bottom
        LCD_DrawFastVLine(x    , y+r  , h-2*r, color); // Left
        LCD_DrawFastVLine(x+w-1, y+r  , h-2*r, color); // Right
        // draw four corners
        LCD_DrawCircleHelper(x+r    , y+r    , r, 1, color);
        LCD_DrawCircleHelper(x+w-r-1, y+r    , r, 2, color);
        LCD_DrawCircleHelper(x+w-r-1, y+h-r-1, r, 4, color);
        LCD_DrawCircleHelper(x+r    , y+h-r-1, r, 8, color);
}

/**
 * \brief Helper function drawing rounded corners
 * 
 * \param x0                        The x-coordinate
 * \param y0                        The y-coordinate
 * \param r                                Radius
 * \param cornername        Corner (1, 2, 3, 4)
 * \param color                        Color
 * 
 * \return void
 */
void LCD_DrawCircleHelper( int16_t x0, int16_t y0, int16_t r, uint8_t cornername, uint16_t color)
{
        int16_t f     = 1 - r;
        int16_t ddF_x = 1;
        int16_t ddF_y = -2 * r;
        int16_t x     = 0;
        int16_t y     = r;

        while (x<y) {
                if (f >= 0) {
                        y--;
                        ddF_y += 2;
                        f     += ddF_y;
                }
                x++;
                ddF_x += 2;
                f     += ddF_x;
                if (cornername & 0x4) {
                        LCD_DrawPixel(x0 + x, y0 + y, color);
                        LCD_DrawPixel(x0 + y, y0 + x, color);
                }
                if (cornername & 0x2) {
                        LCD_DrawPixel(x0 + x, y0 - y, color);
                        LCD_DrawPixel(x0 + y, y0 - x, color);
                }
                if (cornername & 0x8) {
                        LCD_DrawPixel(x0 - y, y0 + x, color);
                        LCD_DrawPixel(x0 - x, y0 + y, color);
                }
                if (cornername & 0x1) {
                        LCD_DrawPixel(x0 - y, y0 - x, color);
                        LCD_DrawPixel(x0 - x, y0 - y, color);
                }
        }
}

/**
 * \brief Draws an circle defined by a pair of coordinates and radius
 * 
 * \param x0                The x-coordinate
 * \param y0                The y-coordinate
 * \param r                        Radius
 * \param color                Color
 * 
 * \return void
 */
void LCD_DrawCircle(int16_t x0, int16_t y0, int16_t r, uint16_t color)
{
        int16_t f = 1 - r;
        int16_t ddF_x = 1;
        int16_t ddF_y = -2 * r;
        int16_t x = 0;
        int16_t y = r;

        LCD_DrawPixel(x0  , y0+r, color);
        LCD_DrawPixel(x0  , y0-r, color);
        LCD_DrawPixel(x0+r, y0  , color);
        LCD_DrawPixel(x0-r, y0  , color);

        while (x<y) {
                if (f >= 0) {
                        y--;
                        ddF_y += 2;
                        f += ddF_y;
                }
                x++;
                ddF_x += 2;
                f += ddF_x;

                LCD_DrawPixel(x0 + x, y0 + y, color);
                LCD_DrawPixel(x0 - x, y0 + y, color);
                LCD_DrawPixel(x0 + x, y0 - y, color);
                LCD_DrawPixel(x0 - x, y0 - y, color);
                LCD_DrawPixel(x0 + y, y0 + x, color);
                LCD_DrawPixel(x0 - y, y0 + x, color);
                LCD_DrawPixel(x0 + y, y0 - x, color);
                LCD_DrawPixel(x0 - y, y0 - x, color);
        }
}

/**
 * \brief Draws a character at the specified coordinates
 * 
 * \param x                        The x-coordinate
 * \param y                        The y-coordinate
 * \param c                        Character
 * \param color                Character color
 * \param bg                Background color
 * \param size                Character Size
 * 
 * \return void
 */
void LCD_DrawChar(int16_t x, int16_t y, unsigned char c, uint16_t color, uint16_t bg, uint8_t size)
{
        if((x >= m_width)            || // Clip right
                (y >= m_height)          || // Clip bottom
                ((x + 6 * size - 1) < 0) || // Clip left
                ((y + 8 * size - 1) < 0))   // Clip top
                        return;

        for (int8_t i=0; i<6; i++ ) {
                uint8_t line;
                if (i == 5) {
                        line = 0x0;
                } else {
                        line = font[c*5 + i];//pgm_read_byte(font+(c*5)+i);
                        for (int8_t j = 0; j<8; j++) {
                                if (line & 0x1) {
                                        if (size == 1) { // default size
                                                LCD_DrawPixel(x+i, y+j, color);
                                        } else {  // big size
                                                LCD_FillRect(x+(i*size), y+(j*size), size, size, color);
                                        }
                                } else if (bg != color) {
                                        if (size == 1) { // default size
                                                LCD_DrawPixel(x+i, y+j, bg);
                                        } else {  // big size
                                                LCD_FillRect(x+i*size, y+j*size, size, size, bg);
                                        }
                                }
                                line >>= 1;
                        }
                }
        }
}

/**
 * \brief Draws a filled circle defined by a pair of coordinates and radius
 * 
 * \param x0                The x-coordinate
 * \param y0                The y-coordinate
 * \param r                        Radius
 * \param color                Color
 * 
 * \return void
 */
void LCD_FillCircle(int16_t x0, int16_t y0, int16_t r, uint16_t color)
{
  LCD_DrawFastVLine(x0, y0-r, 2*r+1, color);
  LCD_FillCircleHelper(x0, y0, r, 3, 0, color);
}

/**
 * \brief Helper function to draw a filled circle
 * 
 * \param x0                        The x-coordinate
 * \param y0                        The y-coordinate
 * \param r                                Radius
 * \param cornername        Corner (1, 2, 3, 4)
 * \param delta                        Delta
 * \param color                        Color
 * 
 * \return void
 */
void LCD_FillCircleHelper(int16_t x0, int16_t y0, int16_t r, uint8_t cornername, int16_t delta, uint16_t color)
{
        int16_t f     = 1 - r;
        int16_t ddF_x = 1;
        int16_t ddF_y = -2 * r;
        int16_t x     = 0;
        int16_t y     = r;

        while (x<y) {
                if (f >= 0) {
                        y--;
                        ddF_y += 2;
                        f     += ddF_y;
                }
                x++;
                ddF_x += 2;
                f     += ddF_x;

                if (cornername & 0x1) {
                        LCD_DrawFastVLine(x0+x, y0-y, 2*y+1+delta, color);
                        LCD_DrawFastVLine(x0+y, y0-x, 2*x+1+delta, color);
                }
                if (cornername & 0x2) {
                        LCD_DrawFastVLine(x0-x, y0-y, 2*y+1+delta, color);
                        LCD_DrawFastVLine(x0-y, y0-x, 2*x+1+delta, color);
                }
        }
}

/**
 * \brief Draws a filled rectangle specified by a coordinate pair, a width, and a height.
 * 
 * \param x                                The x-coordinate of the upper-left corner of the rectangle to draw
 * \param y                                The y-coordinate of the upper-left corner of the rectangle to draw
 * \param w                                Width of the rectangle to draw
 * \param h                                Height of the rectangle to draw
 * \param fillcolor                Color
 * 
 * \return void
 */
void LCD_FillRect(int16_t x, int16_t y1, int16_t w, int16_t h, uint16_t fillcolor)
{
        int16_t  x2, y2;

        // Initial off-screen clipping
        if( (w <= 0) || (h <= 0) ||
                (x >= m_width) || (y1 >= m_height) ||
                ((x2 = x+w-1) <  0) || ((y2  = y1+h-1) <  0))
                        return;
        if(x < 0) { // Clip left
                w += x;
                x = 0;
        }
        if(y1 < 0) { // Clip top
                h += y1;
                y1 = 0;
        }
        if(x2 >= m_width) { // Clip right
                x2 = m_width - 1;
                w  = x2 - x + 1;
        }
        if(y2 >= m_height) { // Clip bottom
                y2 = m_height - 1;
                h  = y2 - y1 + 1;
        }

        LCD_SetAddrWindow(x, y1, x2, y2);
        LCD_Flood(fillcolor, (uint32_t)w * (uint32_t)h);
        LCD_SetAddrWindow(0, 0, m_width - 1, m_height - 1);
}

/**
 * \brief Draws a filled rounded rectangle specified by a coordinate pair, a width, and a height.
 * 
 * \param x                                The x-coordinate of the upper-left corner of the rectangle to draw
 * \param y                                The y-coordinate of the upper-left corner of the rectangle to draw
 * \param w                                Width of the rectangle to draw
 * \param h                                Height of the rectangle to draw
 * \param r                                Radius
 * \param fillcolor                Color
 * 
 * \return void
 */
void LCD_FillRoundRect(int16_t x, int16_t y, int16_t w, int16_t h, int16_t r, uint16_t color)
{
        // smarter version
        LCD_FillRect(x+r, y, w-2*r, h, color);

        // draw four corners
        LCD_FillCircleHelper(x+w-r-1, y+r, r, 1, h-2*r-1, color);
        LCD_FillCircleHelper(x+r    , y+r, r, 2, h-2*r-1, color);
}

/**
 * \brief Fills the screen with the specified color
 * 
 * \param color        Color
 * 
 * \return void
 */
void LCD_FillScreen(uint16_t color)
{
        // For the 932X, a full-screen address window is already the default
        // state, just need to set the address pointer to the top-left corner.
        // Although we could fill in any direction, the code uses the current
        // screen rotation because some users find it disconcerting when a
        // fill does not occur top-to-bottom.
        uint16_t x, y;
        switch(m_rotation) {
                default: x = 0            ; y = 0            ; break;
                case 1 : x = TFTWIDTH  - 1; y = 0            ; break;
                case 2 : x = TFTWIDTH  - 1; y = TFTHEIGHT - 1; break;
                case 3 : x = 0            ; y = TFTHEIGHT - 1; break;
        }
        LCD_CS_LOW();
        LCD_WriteRegister16(0x0020, x);
        LCD_WriteRegister16(0x0021, y);

        LCD_Flood(color, (long)TFTWIDTH * (long)TFTHEIGHT);
}

/**
 * \brief Flood
 * 
 * \param color        Color
 * \param len        Length
 * 
 * \return void
 */
void LCD_Flood(uint16_t color, uint32_t len)
{
        uint16_t blocks;
        uint8_t  i, hi = color >> 8, lo = color;

        LCD_CS_LOW();
        LCD_CD_LOW();
        LCD_Write8(0x00); // High byte of GRAM register...
        LCD_Write8(0x22); // Write data to GRAM

        // Write first pixel normally, decrement counter by 1
        LCD_CD_HIGH();
        LCD_Write8(hi);
        LCD_Write8(lo);
        len--;

        blocks = (uint16_t)(len / 64); // 64 pixels/block
        if(hi == lo) {
                // High and low bytes are identical.  Leave prior data
                // on the port(s) and just toggle the write strobe.
                while(blocks--) {
                        i = 16; // 64 pixels/block / 4 pixels/pass
                        do {
                                LCD_WR_STROBE(); LCD_WR_STROBE(); LCD_WR_STROBE(); LCD_WR_STROBE(); // 2 bytes/pixel
                                LCD_WR_STROBE(); LCD_WR_STROBE(); LCD_WR_STROBE(); LCD_WR_STROBE(); // x 4 pixels
                        } while(--i);
                }
                                // Fill any remaining pixels (1 to 64)
                for(i = (uint8_t)len & 63; i--; ) {
                        LCD_WR_STROBE();
                        LCD_WR_STROBE();
                }
        } else {
                while(blocks--) {
                                i = 16; // 64 pixels/block / 4 pixels/pass
                        do {
                                LCD_Write8(hi); LCD_Write8(lo); LCD_Write8(hi); LCD_Write8(lo);
                                LCD_Write8(hi); LCD_Write8(lo); LCD_Write8(hi); LCD_Write8(lo);
                        } while(--i);
                }
                for(i = (uint8_t)len & 63; i--; ) {
                        LCD_Write8(hi);
                        LCD_Write8(lo);
                }
        }
        LCD_CS_HIGH();
}

/**
 * \brief Print the specified Text
 * 
 * \param fmt        Format text
 * \param 
 * 
 * \return void
 */
void LCD_Printf(const char *fmt, ...)
{
        static char buf[256];
        char *p;
        va_list lst;

        va_start(lst, fmt);
        vsprintf(buf, fmt, lst);
        va_end(lst);

        p = buf;
        while(*p) {
                if (*p == '\n') {
                        m_cursor_y += m_textsize*8;
                        m_cursor_x  = 0;
                } else if (*p == '\r') {
                        // skip em
                } else {
                        LCD_DrawChar(m_cursor_x, m_cursor_y, *p, m_textcolor, m_textbgcolor, m_textsize);
                        m_cursor_x += m_textsize*6;
                        if (m_wrap && (m_cursor_x > (m_width - m_textsize*6))) {
                                m_cursor_y += m_textsize*8;
                                m_cursor_x = 0;
                        }
                }
                p++;
        }
}

/**
 * \brief Resets the Display
 * 
 * \param 
 * 
 * \return void
 */
void LCD_Reset(void)
{
        LCD_CS_HIGH();
        LCD_WR_HIGH();
        LCD_RD_HIGH();

        LCD_RST_LOW();
        delay(100);
        LCD_RST_HIGH();

        // Data transfer sync
        LCD_CS_LOW();

        LCD_CD_LOW();
        LCD_Write8(0x00);
        for(uint8_t i=0; i<3; i++) LCD_WR_STROBE(); // Three extra 0x00s
        LCD_CS_HIGH();
}

/**
 * \brief Sets the cursor coordinates
 * 
 * \param x                The x-coordinate
 * \param y                The y-coordinate
 * 
 * \return void
 */
void LCD_SetCursor(unsigned int x, unsigned int y)
{
        m_cursor_x = x;
        m_cursor_y = y;
}

/**
 * \brief Sets the text size
 * 
 * \param s        Size
 * 
 * \return void
 */
void LCD_SetTextSize(uint8_t s)
{
        m_textsize = (s > 0) ? s : 1;
}

/**
 * \brief Sets the text color
 * 
 * \param c                Text color
 * \param b                Background color
 * 
 * \return void
 */
void LCD_SetTextColor(uint16_t c, uint16_t b)
{
        m_textcolor   = c;
        m_textbgcolor = b;
}

/**
 * \brief Set Text wrap
 * 
 * \param w 
 * 
 * \return void
 */
void LCD_SetTextWrap(uint8_t w)
{
        m_wrap = w;
}

/**
 * \brief Set display rotation
 * 
 * \param x        rotation
 * 
 * \return void
 */
void LCD_SetRotation(uint8_t x)
{
        m_rotation = (x & 3);
        switch(m_rotation) {
        case 0:
        case 2:
                m_width  = TFTWIDTH;
                m_height = TFTHEIGHT;
                break;
        case 1:
        case 3:
                m_width  = TFTHEIGHT;
                m_height = TFTWIDTH;
                break;
        }
}

/**
 * \brief Sets window address
 * 
 * \param x1
 * \param y1
 * \param x2
 * \param y2
 * 
 * \return void
 */
void LCD_SetAddrWindow(int x1, int y1, int x2, int y2)
{

        LCD_CS_LOW();

        // Values passed are in current (possibly rotated) coordinate
        // system.  932X requires hardware-native coords regardless of
        // MADCTL, so rotate inputs as needed.  The address counter is
        // set to the top-left corner -- although fill operations can be
        // done in any direction, the current screen rotation is applied
        // because some users find it disconcerting when a fill does not
        // occur top-to-bottom.
        int x, y, t;
        switch(m_rotation) {
        default:
                x  = x1;
                y  = y1;
                break;
        case 1:
                t  = y1;
                y1 = x1;
                x1 = TFTWIDTH  - 1 - y2;
                y2 = x2;
                x2 = TFTWIDTH  - 1 - t;
                x  = x2;
                y  = y1;
                break;
        case 2:
                t  = x1;
                x1 = TFTWIDTH  - 1 - x2;
                x2 = TFTWIDTH  - 1 - t;
                t  = y1;
                y1 = TFTHEIGHT - 1 - y2;
                y2 = TFTHEIGHT - 1 - t;
                x  = x2;
                y  = y2;
                break;
        case 3:
                t  = x1;
                x1 = y1;
                y1 = TFTHEIGHT - 1 - x2;
                x2 = y2;
                y2 = TFTHEIGHT - 1 - t;
                x  = x1;
                y  = y2;
                break;
        }

        LCD_WriteRegister16(0x0050, x1); // Set address window
        LCD_WriteRegister16(0x0051, x2);
        LCD_WriteRegister16(0x0052, y1);
        LCD_WriteRegister16(0x0053, y2);
        LCD_WriteRegister16(0x0020, x ); // Set address counter to top left
        LCD_WriteRegister16(0x0021, y );

        LCD_CS_HIGH();
}

/**
 * \brief Writes 8-Bit data
 * 
 * \param data        8-Bit Data
 * 
 * \return void
 */
void LCD_Write8(uint8_t data)
{
        // ------ PORT -----     --- Data ----
        // GPIOA, GPIO_PIN_9  -> BIT 0 -> 0x01
        // GPIOC, GPIO_PIN_7  -> BIT 1 -> 0x02
        // GPIOA, GPIO_PIN_10 -> BIT 2 -> 0x04
        // GPIOB, GPIO_PIN_3  -> BIT 3 -> 0x08
        // GPIOB, GPIO_PIN_5  -> BIT 4 -> 0x10
        // GPIOB, GPIO_PIN_4  -> BIT 5 -> 0x20
        // GPIOB, GPIO_PIN_10 -> BIT 6 -> 0x40
        // GPIOA, GPIO_PIN_8  -> BIT 7 -> 0x80

        GPIOA->ODR = (GPIOA->ODR & 0xF8FF) |
                        ((data & 0x01) << 9) | ((data & 0x04) << 8) | ((data & 0x80) << 1);
        GPIOB->ODR = (GPIOB->ODR & 0xFBC7) |
                        (data & 0x08) | ((data & 0x10) << 1) | ((data & 0x20) >> 1) | ((data & 0x40) << 4);
        GPIOC->ODR = (GPIOC->ODR & 0xFF7F) | ((data & 0x02) << 6);

        LCD_WR_STROBE();
}

/**
 * \brief Writes 8-Bit register
 * 
 * \param data        8-Bit Data
 * 
 * \return void
 */
void LCD_WriteRegister8(uint8_t a, uint8_t d)
{
        LCD_CD_LOW();
        LCD_Write8(a);
        LCD_CD_HIGH();
        LCD_Write8(d);
}

/**
 * \brief Writes 16-Bit register
 * 
 * \param a                Register
 * \param d                Data
 * 
 * \return void
 */
void LCD_WriteRegister16(uint16_t a, uint16_t d)
{
        uint8_t hi, lo;
        hi = (a) >> 8;
        lo = (a);
        LCD_CD_LOW();
        LCD_Write8(hi);
        LCD_Write8(lo);
        hi = (d) >> 8;
        lo = (d);
        LCD_CD_HIGH();
        LCD_Write8(hi);
        LCD_Write8(lo);
}
