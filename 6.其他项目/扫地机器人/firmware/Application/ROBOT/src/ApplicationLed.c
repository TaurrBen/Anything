#include "applicationled.h"

/*
 * Filename   ��applicationled.c
 * Author     : pusheng       
 * Version    : 1.0
 * Date       : 2019.11.7
 * Discription : LED����Ӧ�ú���
*/	

/*******************************************************************************
* Function Name  : vLedStateDisplay
* Description    : LED��ʾ
* Input          : None
* Output         : None
* Return         : None
*******************************************************************************/
void vLedStateDisplay(void)
{
    static u8 Bright=0,dir=0;
    static OS_TimeStruct BrigTime={0};//�֡��롢���룬�ñ������ڳ�ʱ��⺯���и���Ϊ��ǰֵ
    
    if(TimeOut == MultipleTimeoutCheck(&BrigTime,0,0,5))
    {
        if(Bright > 100) dir=1;
        else
        if(Bright == 0) dir=0;
            
        if(dir) Bright--;   //����ʱ���𽥱䰵
        else    Bright++;   //��ȫϨ��ʱ���𽥱���
        
		//case1���ڳ�����ϳ�磨ȫ������case2���ڳ������û�г�磨��ɫ������case3��û���ڳ�����ϣ���ɫ����
        if(isInRechargeStand())//�ڳ������
        {
            if(usGetChargeCurrent()>50)//�г��
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
* Description    : ��ɫLED��Ӧ������ʾ  
* Input          : Bright 0-100  Խ��Խ�� 
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
* Description    : ��ɫLED��Ӧ������ʾ 
* Input          : Bright 0-100  Խ��Խ�� 
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
* Description    : ��ɫLED��Ӧ������ʾ  
* Input          : Bright 0-100  Խ��Խ�� 
* Output         : None
* Return         : None
*******************************************************************************/
void vYellowLedBrightness(u8 Bright)
{
    if(Bright > 100)
        Bright=100;
    
    YELLOWLED_PWM_CYCLE(Bright);  
}

