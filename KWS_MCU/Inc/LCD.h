#ifndef LCD_H
#define LCD_H

#include <stdint.h>

#define SUPPORT_ECRAN_3

#define	BLACK   0x0000
#define	BLUE    0x001F
#define	RED     0xF800
#define	GREEN   0x07E0
#define CYAN    0x07FF
#define MAGENTA 0xF81F
#define YELLOW  0xFFE0
#define WHITE   0xFFFF
#define LIGHTGRAY 0xCDB6


void LCD_Begin(void);
void LCD_DrawPixel(int16_t x, int16_t y, uint16_t color);
void LCD_DrawLine(int16_t x1, int16_t y1, int16_t x2, int16_t y2, uint16_t color);
void LCD_DrawFastHLine(int16_t x, int16_t y, int16_t length, uint16_t color);
void LCD_DrawFastVLine(int16_t x, int16_t y, int16_t h, uint16_t color);
void LCD_DrawRect(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t color);
void LCD_DrawRoundRect(int16_t x, int16_t y, int16_t w, int16_t h, int16_t r, uint16_t color);
void LCD_DrawCircleHelper( int16_t x0, int16_t y0, int16_t r, uint8_t cornername, uint16_t color);
void LCD_DrawCircle(int16_t x0, int16_t y0, int16_t r, uint16_t color);
void LCD_DrawChar(int16_t x, int16_t y, unsigned char c, uint16_t color, uint16_t bg, uint8_t size);
void LCD_FillCircle(int16_t x0, int16_t y0, int16_t r, uint16_t color);
void LCD_FillCircleHelper(int16_t x0, int16_t y0, int16_t r, uint8_t cornername, int16_t delta, uint16_t color);
void LCD_FillRect(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t fillcolor);
void LCD_FillRoundRect(int16_t x, int16_t y, int16_t w, int16_t h, int16_t r, uint16_t color);
void LCD_FillScreen(uint16_t color);
void LCD_Flood(uint16_t color, uint32_t len);
void LCD_Reset(void);
void LCD_SetCursor(unsigned int x, unsigned int y);
void LCD_SetTextSize(uint8_t s);
void LCD_SetTextColor(uint16_t c, uint16_t b);
void LCD_SetTextWrap(uint8_t w) ;
void LCD_SetRotation(uint8_t x);
void LCD_SetAddrWindow(int x1, int y1, int x2, int y2);
void LCD_Printf(const char *fmt, ...);

void LCD_Write8(uint8_t data);
void LCD_WriteRegister8(uint8_t a, uint8_t d);
void LCD_WriteRegister16(uint16_t a, uint16_t d);

uint16_t LCD_Color565(uint8_t r, uint8_t g, uint8_t b);

#endif /* LCD_H */
