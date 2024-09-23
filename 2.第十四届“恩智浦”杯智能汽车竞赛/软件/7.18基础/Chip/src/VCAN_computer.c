/*!
 *     COPYRIGHT NOTICE
 *     Copyright (c) 2015,ɽ��Ƽ�
 *     All rights reserved.
 *     �������ۣ�ɽ����̳ http://www.vcan123.com
 *
 *     ��ע�������⣬�����������ݰ�Ȩ����ɽ��Ƽ����У�δ����������������ҵ��;��
 *     �޸�����ʱ���뱣��ɽ��Ƽ��İ�Ȩ������
 *
 * @file       VCAN_computer.c
 * @brief      ɽ��๦�ܵ���������λ����������
 * @author     ɽ��Ƽ�
 * @version    v5.2.2
 * @date       2015-03-24
 */
  

/*
 * ����ͷ�ļ�
 */
#include "common.h"
#include "MK60_uart.h"
#include "VCAN_computer.h"
#include "include.h"

extern uint8 sending_flag;

//extern uint8 DMA_Over_Flg;
//extern uint8 img_mix_data[CAMERA_H][CAMERA_W];
/*!
 *  @brief      ɽ��๦�ܵ���������λ��������ͷ��ʾ����
 *  @param      imgaddr    ͼ����ʼ��ַ
 *  @param      imgsize    ͼ��ռ�ÿռ�Ĵ�С
 *  @since      v5.0
*  Sample usage:
             �����÷��ο�������:
            ��ɽ������ͷ��ӥ����λ�����̺�΢��Ч�� - ���ܳ�������
             http://vcan123.com/forum.php?mod=viewthread&tid=6242&ctid=27
 */
#define VCAN_POR UART0
void vcan_sendimg(void *imgaddr, uint32_t imgsize)
{
#define CMD_IMG     1
    uint8_t cmdf[2] = {CMD_IMG, ~CMD_IMG};    //ɽ����λ�� ʹ�õ�����
    uint8_t cmdr[2] = {~CMD_IMG, CMD_IMG};    //ɽ����λ�� ʹ�õ�����
    uart_putbuff(VCAN_POR, cmdf, sizeof(cmdf));    //�ȷ�������
    uart_putbuff(VCAN_POR, (uint8_t *)imgaddr, imgsize); //�ٷ���ͼ
    uart_putbuff(VCAN_POR, cmdr, sizeof(cmdr));    //�ȷ�������
	sending_flag=0;
}
void mat_sendimg(void *imgaddr, uint32_t imgsize)
{
    uint8_t ff[1]={255};
    uart_putbuff(VCAN_POR, ff, sizeof(ff));
    uart_putbuff(VCAN_POR, (uint8_t *)imgaddr, imgsize); //�ٷ���ͼ��
}
extern void vcan_hsw(void *imgaddr, uint32_t imgsize){
     uart_putchar(VCAN_POR, 0xff);
     uart_putbuff(VCAN_POR, (uint8_t *)imgaddr, imgsize); //�ٷ���ͼ��
     
}
void vcan_sendware(void *wareaddr, uint32_t waresize)
{
#define CMD_WARE     3
    uint8_t   cmdf[2] = {CMD_WARE, ~CMD_WARE};    //���ڵ��� ʹ�õ�ǰ����
    uint8_t cmdr[2] = {~CMD_WARE, CMD_WARE};    //���ڵ��� ʹ�õĺ�����

    uart_putbuff(VCAN_PORT, cmdf, sizeof(cmdf));    //�ȷ���ǰ����
    uart_putbuff(VCAN_PORT, (uint8_t *)wareaddr, waresize);    //��������
    uart_putbuff(VCAN_PORT, cmdr, sizeof(cmdr));    //���ͺ�����

}


//void sendimage()
//{
//    if(DMA_Over_Flg==1)     //����ͼ����λ��
//    {
//     DisableInterrupts;         //�ر��ж�
////      uart_putchar(UART4,0xff);
//      uint8 h,l;
//      for(h=0;h<CAMERA_H;h++)
//      {
//        for(l=0;l<CAMERA_W;l++)
//        {
//          if(img_mix_data[h][l] == 0xff)
//            img_mix_data[h][l] = 0xfe;
//          uart_putchar(UART4,img_mix_data[h][l]);
//        }
//      }
//      uart_putchar(UART4,0xff);
//      DMA_Over_Flg=0;
//      EnableInterrupts  ;  //�����ж�
//    }
//}