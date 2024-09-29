#include "applicationadc.h"

/*
 * Filename   ：applicationadc.c
 * Author     : pusheng       
 * Version    : 1.0
 * Date       : 2019.11.12
 * Discription : ADC数据处理应用函数
*/	

TaskHandle_t HandleADCDataProcessing = NULL;//adc数据处理任务句柄

uint16_t g_usInChipAdcChannelAverageValue[ADC_CHANNEL_NUM]; //ADC 各通道平均值（总13个通道）   
uint16_t g_usAnalogSwitch[adcAnalog_Switch_Num];            //模拟开关ADC采样值（4个）

/*******************************************************************************
* Function Name  : vADCDataProcessingTask
* Description    : ADC数据处理任务
* Input          : p 传入参数
* Output         : None
* Return         : None
*******************************************************************************/
void vADCDataProcessingTask(void *p)
{
    ADC_SoftwareStartConvCmd(ADC1, ENABLE);	  //使能指定的ADC1的软件转换启动功能
    while(1)
    {
        if( ulTaskNotifyTake( pdFALSE , portMAX_DELAY) )//任务会等待一个通知，阻塞等待adc转换完成
        {
            vGetADCChannelAverageNumber( g_usInChipAdcChannelAverageValue, g_usAnalogSwitch );//数据处理
            ADC_Cmd( ADC1, ENABLE );//ADON位
            ADC_SoftwareStartConvCmd(ADC1, ENABLE);//SWSTART位，单次转换模式下每次转换都要向这位写1
        }
    }
}


/*******************************************************************************
* Function Name  : vGetADCChannelAverageNumber
* Description    : 获取ADC每个通道采集值
* Input          : None
* Output         : *pucBuffer：用于装载ADC每个通道平局值数组,pusSwitch：模拟开关ADC值
* Return         : None
*******************************************************************************/
void vGetADCChannelAverageNumber( uint16_t *pusBuffer, uint16_t *pusSwitch ) 
{
    #define SELECT_ALONG_EDGE_ADC       SGM4878_A=0;SGM4878_B=0; ucAnalogPoint=adcAlong_Edge;   //沿边ADC，编号0
    #define SELECT_BATTERY_ID_ADC       SGM4878_A=1;SGM4878_B=0; ucAnalogPoint=adcBatteryID;    //电池ID ADC，编号1
    #define SELECT_CHARGE_CURRENT_ADC   SGM4878_A=0;SGM4878_B=1; ucAnalogPoint=adcChargeCurrent;//充电电流ADC，编号2
    #define SELECT_RECHARGES_STAND_ADC  SGM4878_A=1;SGM4878_B=1; ucAnalogPoint=adcRechargeStand;//充电桩ADC，编号3

    static uint8_t ucAnalogPoint = 0;     //usAnalogSwitch[]下标，静态类型
    int8_t i,j,k;
    uint32_t uiSum[ADC_CHANNEL_NUM] = {0};//长度为13的u32数组，保存每个通道所有采样值的和
    uint16_t AxtractData[ADC_CHANNEL_NUM][ADC_TRANSFORM_NUM] = {0};//13×3的二维数组，使用DMA转换的结果进行填充
    uint16_t usTemp;
    
    /*****************取出每个ADC通道的值 ***************/
    for(i=0;i<ADC_TRANSFORM_NUM;i++)    //转换次数
    {
        for(j=0;j<ADC_CHANNEL_NUM;j++)  //转换通道
        {
            AxtractData[j][i] = g_AdcOriginalValue[i][j];//3×13->13×3
        }        
    }
    
    /***************************插入排序算法****************************/
    for( k=0;k<ADC_CHANNEL_NUM;k++ )      //通道
    {
        for( i=1;i<ADC_TRANSFORM_NUM;i++ )//转换次数
        { 
            usTemp=AxtractData[k][i];                         //usTemp为插入的元素
            j=i-1; 
            while( ( j>=0 ) && ( usTemp<AxtractData[k][j] ) ) //查找array[i-1] 后比它小的数
            { 
                AxtractData[k][j+1]=AxtractData[k][j]; 
                j--; 
            } 
            AxtractData[k][j+1]=usTemp;                       //插入数据
        } 
    }
      
    /******************计算每个ADC通道平均值******************/
    for( i=0;i<ADC_CHANNEL_NUM;i++ )        
    {
        for( j=1;j<ADC_TRANSFORM_NUM-1;j++ )          //去掉一个最大值和最小值
        {
            uiSum[i] += AxtractData[i][j];            //每个通道的所有转换值相加
        }         
        pusBuffer[i] = uiSum[i]/(ADC_TRANSFORM_NUM-2);//求每个通道的平均值
    }

    /**************取出模拟开关对应通道的ADC值***************/
    switch( ucAnalogPoint )
    {
        case adcAlong_Edge :
            pusSwitch[adcAlong_Edge] = pusBuffer[adcAnalogChannel];
            SELECT_BATTERY_ID_ADC;       //选择下一个通道                
            break;                                    
        case adcBatteryID :
            pusSwitch[adcBatteryID] = pusBuffer[adcAnalogChannel];
            SELECT_CHARGE_CURRENT_ADC; //选择下一个通道
           break;            
        case adcChargeCurrent :
            pusSwitch[adcChargeCurrent] = pusBuffer[adcAnalogChannel];
            SELECT_RECHARGES_STAND_ADC; //选择下一个通道               
            break;            
        case adcRechargeStand :
            pusSwitch[adcRechargeStand] = pusBuffer[adcAnalogChannel];
            SELECT_ALONG_EDGE_ADC; //选择下一个通道               
            break; 
        default: break;
    }           
    
}


