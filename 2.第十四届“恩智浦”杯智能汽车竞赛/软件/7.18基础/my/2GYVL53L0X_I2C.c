#include "common.h"
#include "include.h"//GYVL53L0X
#include "2GYVL53L0X_I2C.h"
#define i2c_delay() IIC_Delay(10)
void IIC_Delay(uint16 i)
{	
  while(i) 
    i--;
}
void GYVL53L0X_Read_Rev(uint8 * RevID)
{
    *RevID =  i2c_read_reg(GYVL53L0X_I2C,GYVL53L0X_ADDRESS,VL53L0X_REG_IDENTIFICATION_REVISION_ID); 
}
void GYVL53L0X_Read_DevID(uint8 * DevID)
{
    *DevID =  i2c_read_reg(GYVL53L0X_I2C,GYVL53L0X_ADDRESS,VL53L0X_REG_IDENTIFICATION_MODEL_ID); 
}

void GYVL53L0X_Read_data(uint8 addr,uint16 * dist,uint16 * data1,uint16 * data2,uint16 * data3)
{
    I2C_WriteReg( 0X29, 0X00, 0x01);
    i2c_delay();
    uint8 Read_GYVL53L0X_Flag = 1;
    uint16 time = 0;
    while(Read_GYVL53L0X_Flag)
    {
        uint8 temp = I2C_ReadByte(addr,VL53L0X_REG_RESULT_RANGE_STATUS);
        i2c_delay();
        if(temp&0x01){Read_GYVL53L0X_Flag = 2;break;}else{time++;}
        if(time>=6)break;
    }
    uint8 data[12];
    for(uint8 i=0;i<12;i++)
    {
        data[i] = I2C_ReadByte(addr,(0x14+i));
        DELAY_MS(1);
    }
    //da1 = (data[0]& 0x78) >> 3;
    //*data0 = (data[2]<<8) + data[3];//Effective Spad Rtn Count format 8,8
    *data1 = (data[4]<<8) + data[5];
    *data2 = (data[6]<<8) + data[7];//Signal Rate format 9,7
    *data3 = (data[8]<<8) + data[9];//Ambient Rate format 9,7
    *dist = (data[10]<<8) + data[11];
    
    
    if(*data1<100||*data1>2000) *data1=0;
    if(*data2<50||*data2>2000) *data2=0;
    if(*data3<100||*data3>2500) *data3=0; 
    Read_GYVL53L0X_Flag = 0;
}