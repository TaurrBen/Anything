#ifndef OLED_H_H
#define OLED_H_H
 
#define byte uint8
#define word uint16
#define GPIO_PIN_MASK      0x1Fu    //0x1f=31,����λ��Ϊ0--31��Ч
#define GPIO_PIN(x)        (((1)<<(x & GPIO_PIN_MASK)))  //�ѵ�ǰλ��1

#define LCD_Port_Init()            gpio_init (PTD3, GPO,0);   \
                                   gpio_init (PTD0, GPO,0);   \
                                   gpio_init (PTC12, GPO,0);   \
                                   gpio_init (PTC18, GPO,0);   \
                                   gpio_init (PTC19, GPO,0);  \
                                   gpio_init (PTD1, GPO,0);  
#define LCD_DC_HIGH     gpio_set(PTC18, 1)//c18   c12
#define LCD_DC_LOW      gpio_set(PTC18, 0)

#define LCD_SCL_HIGH    gpio_set(PTD1, 1)//D1     D3
#define LCD_SCL_LOW     gpio_set(PTD1, 0)

#define LCD_SDA_HIGH    gpio_set(PTD0, 1)
#define LCD_SDA_LOW     gpio_set(PTD0, 0)

#define LCD_RST_HIGH    gpio_set(PTC19, 1)//c19  C18
#define LCD_RST_LOW     gpio_set(PTC19, 0)



 extern byte longqiu96x64[768];
 void Dly_ms(uint16 ms);
 void LCD_Init(void);
 void LCD_CLS(void);
 void LCD_Set_Pos(byte x, byte y);
 void LCD_WrDat(byte data);
 void LCD_P6x8Str(byte x,byte y,byte ch[]);//д��һ���׼ASCII�ַ������ַ�ռ6x8
 void LCD_P8x16Str(byte x,byte y,byte ch[]);//д��һ���׼ASCII�ַ�����һ�ַ�����ռ8���ص㣬����ռ16���ص�
 void LCD_P14x16Str(byte x,byte y,byte ch[]);//��������ַ���
 void LCD_Print(byte x, byte y, byte ch[]);//������ֺ��ַ�����ַ���
 void LCD_PutPixel(byte x,byte y);//������һ���㣨x,y��
 void LCD_Rectangle(byte x1,byte y1,byte x2,byte y2,byte gif);//����һ��ʵ�ľ���
 void Draw_BMP(byte x0,byte y0,byte x1,byte y1,byte bmp[]); //��ʾBMPͼƬ128��64
 void LCD_Fill(byte dat);
 void Dis_SNum(byte x, byte y, uint16 num,byte N) ;
 void Dis_Num(byte x, byte y, uint16 num,byte N) ;
 void Dis_Float(byte X,byte Y,double real,byte N);
 void Dis_Float2(byte X,byte Y,double real,byte N1,byte N2);
 void OledPrint(uint16_t ADdataCount,uint16_t LineCount,uint16_t PredatorFAV, uint8 *dataAD); //������ʾ
 void LED_PrintImage(uint16 usColumnNum,uint16 usRowNum,uint8 *pucTable);
 void OLED_IMAGE(uint8 (*tempimg)[80],uint8 *Certre_Line);
 void CCD_IMG(uint8 *ccd,uint8 *Thread);
 void LCD_ClrDot(unsigned char x) ; //����CCDʹ��
 void OLED_IMAGE_OV7725(uint8 (*data)[80],int16 *Certre_Line);
 void MY_OLED_IMAGE(uint8 (*tempimg)[160]);

 
#endif