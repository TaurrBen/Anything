#include "applicationled.h"

/*
 * Filename   ：applicationled.c
 * Author     : pusheng       
 * Version    : 1.0
 * Date       : 2019.11.7
 * Discription : LED处理应用函数
*/	

/*******************************************************************************
* Function Name  : vLedStateDisplay
* Description    : LED显示
* Input          : None
* Output         : None
* Return         : None
*******************************************************************************/
void vLedStateDisplay(void)
{
    static u8 Bright=0,dir=0;
    static OS_TimeStruct BrigTime={0};//分、秒、毫秒，该变量会在超时检测函数中更新为当前值
    
    if(TimeOut == MultipleTimeoutCheck(&BrigTime,0,0,5))
    {
        if(Bright > 100) dir=1;
        else
        if(Bright == 0) dir=0;
            
        if(dir) Bright--;   //最亮时，逐渐变暗
        else    Bright++;   //完全熄灭时，逐渐变亮
        
		//case1：在充电座上充电（全亮），case2：在充电座上没有充电（黄色亮），case3：没有在充电座上（白色亮）
        if(isInRechargeStand())//在充电座上
        {
            if(usGetChargeCurrent()>50)//有充电
            {
                vRedLedBrightness( Bright);
                vWhiteLedBrightness( Bright);
                vYellowLedBrightness( Bright);
            }
			else
            {
                vRedLedBrightness( 0);
                vWhiteLedBrightness( 0);
                vYellowLedBrightness( Bright);
            }
        }
		else
        {
            vRedLedBrightness( 0);
            vWhiteLedBrightness( Bright);
            vYellowLedBrightness( 0);
        }
    }
}

/*******************************************************************************
* Function Name  : vRedLedBrightness(u8 Bright)
* Description    : 红色LED对应亮度显示  
* Input          : Bright 0-100  越大越亮 
* Output         : None
* Return         : None
*******************************************************************************/
void vRedLedBrightness(u8 Bright)
{
    if(Bright > 100)
        Bright=100;
    
    REDLED_PWM_CYCLE(Bright);  
}
/*******************************************************************************
* Function Name  : vWhiteLedBrightness(u8 Bright)
* Description    : 白色LED对应亮度显示 
* Input          : Bright 0-100  越大越亮 
* Output         : None
* Return         : None
*******************************************************************************/
void vWhiteLedBrightness(u8 Bright)
{
    if(Bright > 100)
        Bright=100;
    
    WHITELED_PWM_CYCLE(Bright);  
}
/*******************************************************************************
* Function Name  : vYellowLedBrightness(u8 Bright)
* Description    : 黄色LED对应亮度显示  
* Input          : Bright 0-100  越大越亮 
* Output         : None
* Return         : None
*******************************************************************************/
void vYellowLedBrightness(u8 Bright)
{
    if(Bright > 100)
        Bright=100;
    
    YELLOWLED_PWM_CYCLE(Bright);  
}

