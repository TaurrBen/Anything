#ifndef _I2C_H
#define _I2C_H
#include "include.h"


//io²Ù×÷º¯Êý

#define USERI2C_SCL    	        gpio_ddr (PTB0, GPO);      //SCL  
#define USERI2C_SDA 	        gpio_ddr (PTB1, GPO);      //SDA	 
#define USERI2C_READ_SDA   	gpio_ddr (PTB1, GPI);      //ÊäÈëSDA 


void UserI2c_Init(void); 
void IIC_Start(void);
void IIC_Stop(void);	
void UserI2c_Send_Byte(unsigned char dat);
uint8 UserI2c_Read_Byte(void);
uint8 IIC_Read_Byte(unsigned char ack);
uint8 IIC_Wait_Ack(void); 				
void IIC_Ack(void);					
void IIC_NAck(void);


uint8 SensorWritenByte(unsigned char *txbuff, unsigned char regaddr, unsigned char size);
uint8 SensorReadnByte(unsigned char *rtxbuff, unsigned char regaddr, unsigned char size);

void Sensor_I2C_Test(void);

#endif