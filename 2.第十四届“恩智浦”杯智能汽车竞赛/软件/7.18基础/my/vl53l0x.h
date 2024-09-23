#ifndef _VL53L0_H
#define _VL53L0_H
#include "include.h"


uint8 VL53L0X_Write_Len(uint8 addr,uint8 reg,uint8 len,uint8 *buf);

uint8 VL53L0X_Read_Len(uint8 addr,uint8 reg,uint8 len,uint8 *buf);

uint8 VL53L0X_Write_Byte(uint8 reg,uint8 data);

uint8 VL53L0X_Read_Byte(uint8 reg);
