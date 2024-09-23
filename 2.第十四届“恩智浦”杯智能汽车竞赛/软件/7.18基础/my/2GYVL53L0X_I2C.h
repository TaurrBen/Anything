#ifndef _GYVL53L0X_I2C_H_
#define _GYVL53L0X_I2C_H_

#define VL53L0X_REG_SYSTEM_THRESH_HIGH              0x0C//阈值高
#define VL53L0X_REG_SYSTEM_THRESH_LOW               0x0E//阈值低

#define VL53L0X_REG_SYSTEM_SEQUENCE_CONFIG		    0x01//时序配置 
#define VL53L0X_REG_SYSTEM_RANGE_CONFIG			    0x09//范围配置
#define VL53L0X_REG_SYSTEM_INTERMEASUREMENT_PERIOD	0x04//互测周期 

#define VL53L0X_REG_SYSTEM_INTERRUPT_CONFIG_GPIO    0x0A//中断GPIO高速模式需要
/*
#define VL53L0X_REG_SYSTEM_INTERRUPT_GPIO_DISABLED	0x00
#define VL53L0X_REG_SYSTEM_INTERRUPT_GPIO_LEVEL_LOW	0x01
#define VL53L0X_REG_SYSTEM_INTERRUPT_GPIO_LEVEL_HIGH	0x02
#define VL53L0X_REG_SYSTEM_INTERRUPT_GPIO_OUT_OF_WINDOW	0x03
#define VL53L0X_REG_SYSTEM_INTERRUPT_GPIO_NEW_SAMPLE_READY	0x04
*/
#define VL53L0X_REG_SYSTEM_INTERRUPT_CLEAR          0x0B

#define VL53L0X_REG_GPIO_HV_MUX_ACTIVE_HIGH         0x84




#define VL53L0X_REG_IDENTIFICATION_MODEL_ID         0xc0
#define VL53L0X_REG_IDENTIFICATION_REVISION_ID      0xc2
#define VL53L0X_REG_PRE_RANGE_CONFIG_VCSEL_PERIOD   0x50
#define VL53L0X_REG_FINAL_RANGE_CONFIG_VCSEL_PERIOD 0x70
#define VL53L0X_REG_SYSRANGE_START                  0x00
/* VL53L0X_REG_SYSRANGE_START
bit 0 write 1 toggle state in continuous mode and arm next shot in single shot mode 
bit 1 write 0 set single shot mode
bit 1 write 1 set back-to-back operation mode
bit 2 write 1 set timed operation mode
bit 3 write 1 set histogram operation mode
*/
#define VL53L0X_REG_RESULT_INTERRUPT_STATUS         0x13
#define VL53L0X_REG_RESULT_RANGE_STATUS             0x14
#define GYVL53L0X_ADDRESS  0x29
#define GYVL53L0X_I2C I2C1
extern void GYVL53L0X_Read_Rev(uint8 * RevID);
extern void GYVL53L0X_Read_DevID(uint8 * DevID);
extern void GYVL53L0X_Read_data(uint8 addr,uint16 * dist,uint16 * data1,uint16 * data2,uint16 * data3);
extern void IIC_Delay(uint16 i);
/**
 * @struct VL53L0X_RangeData_t
 * @brief Range measurement data.
 */
typedef struct {
	uint32_t TimeStamp;		/*!< 32-bit time stamp. */
	uint32_t MeasurementTimeUsec;
		/*!< Give the Measurement time needed by the device to do the
		 * measurement.*/


	uint16_t RangeMilliMeter;	/*!< range distance in millimeter. */

	uint16_t RangeDMaxMilliMeter;
		/*!< Tells what is the maximum detection distance of the device
		 * in current setup and environment conditions (Filled when
		 *	applicable) */

	uint32_t SignalRateRtnMegaCps;
		/*!< Return signal rate (MCPS)\n these is a 16.16 fix point
		 *	value, which is effectively a measure of target
		 *	 reflectance.*/
	uint32 AmbientRateRtnMegaCps;
		/*!< Return ambient rate (MCPS)\n these is a 16.16 fix point
		 *	value, which is effectively a measure of the ambien
		 *	t light.*/

	uint16_t EffectiveSpadRtnCount;
		/*!< Return the effective SPAD count for the return signal.
		 *	To obtain Real value it should be divided by 256 */

	uint8_t ZoneId;
		/*!< Denotes which zone and range scheduler stage the range
		 *	data relates to. */
	uint8_t RangeFractionalPart;
		/*!< Fractional part of range distance. Final value is a
		 *	FixPoint168 value. */
	uint8_t RangeStatus;
		/*!< Range Status for the current measurement. This is device
		 *	dependent. Value = 0 means value is valid.
		 *	See \ref RangeStatusPage */
} VL53L0X_RangingMeasurementData_t;




#endif //_GYVL53L0X_I2C_H_