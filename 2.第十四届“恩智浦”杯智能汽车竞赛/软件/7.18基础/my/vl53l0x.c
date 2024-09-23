#include "include.h"


//uint8 VL53L0X_Write_Byte(uint8 reg,uint8 data) 				 
//{ 
//        i2c_Start(I2C0); 
//	i2c_write_byte	 i2c_write_reg(I2C0, 0x1D, 1,2);
//	if(IIC_Wait_Ack())	
//	{
//		IIC_Stop();		 
//		return 1;		
//	}
//    IIC_Send_Byte(reg);	
//    IIC_Wait_Ack();		
//	IIC_Send_Byte(data);
//	if(IIC_Wait_Ack())	
//	{
//		IIC_Stop();	 
//		return 1;		 
//	}		 
//    IIC_Stop();	 
//	return 0;
//}