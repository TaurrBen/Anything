#include "include.h"
void My_IIC_Port_Init()
{
  gpio_init(I2C_SCL_PORT,GPO,1);                     //��ʼ��SCL�ܽ�
  gpio_init(I2C_SDA_PORT,GPO,1);                     //��ʼ��SDA�ܽ�
  port_init_NoALT (I2C_SCL_PORT, PULLUP );
  port_init_NoALT (I2C_SDA_PORT, PULLUP );
}
/*
 * I2C_WriteReg
 * дI2C�豸�Ĵ���
 */
void I2C_WriteReg(uint8 dev_addr,uint8 reg_addr , uint8 data)
{		
  I2C_Start();
  I2C_SendByte(dev_addr<<1);         
  I2C_SendByte(reg_addr );   
  I2C_SendByte(data);   
  I2C_Stop(); 
}
//���Ĵ���
uint8 I2C_ReadByte(uint8 dev_addr,uint8 reg_addr)
{
  uint8 data;
  I2C_Start();
  I2C_SendByte( dev_addr<<1);//�ӻ���ַ+д�ź�
  I2C_SendByte( reg_addr );//�Ĵ�����ַ
  I2C_Start();
  I2C_SendByte((dev_addr<<1)+1);//�ӻ���ַ+���ź�
  data= I2C_ReceiveByte();
  I2C_NoAck();
  I2C_Stop();
  return data;
}

//���Ĵ���
int16 I2C_ReadWord(uint8 dev_addr,uint8 reg_addr)
{
  uint8 h,l;
  I2C_Start();
  I2C_SendByte( dev_addr<<1); 
  I2C_SendByte( reg_addr);
  I2C_Start();
  I2C_SendByte((dev_addr<<1)+1);
  h= I2C_ReceiveByte();
  I2C_Ack();
  l= I2C_ReceiveByte();
  I2C_NoAck();
  I2C_Stop();
  return (h<<8)+l;
}

//�����������������������
void I2C_ReadGryo(uint8 dev_addr,uint8 reg_addr,int16 *x,int16 *y)
{
  char h,l;
  I2C_Start();
  I2C_SendByte( dev_addr<<1); 
  I2C_SendByte( reg_addr); 
  I2C_Start();	
  I2C_SendByte((dev_addr<<1)+1); 
  h= I2C_ReceiveByte();
  I2C_Ack();
  l= I2C_ReceiveByte();
  I2C_Ack();
  *x=(h<<8)+l;
  h= I2C_ReceiveByte();
  I2C_Ack();
  l= I2C_ReceiveByte();
  I2C_Ack();
  h= I2C_ReceiveByte();
  I2C_Ack();
  l= I2C_ReceiveByte();
  I2C_NoAck();
  *y=(h<<8)+l;
  I2C_Stop();
}
/*
 * I2C_Start
 * I2C��ʼ�źţ��ڲ�����
 */
void I2C_Start(void)
{
  I2C_SDA_OUT();
  I2C_DELAY();
  I2C_DELAY();
  I2C_SDA_O=1; 
  I2C_SCL=1;
  I2C_DELAY();
  I2C_DELAY();
  I2C_SDA_O=0; 
  I2C_DELAY();
  I2C_DELAY();
  I2C_SCL=0;
  I2C_DELAY();
  I2C_DELAY();
}

/*
 * I2C_Stop
 * I2Cֹͣ�źţ��ڲ�����
 */
 void I2C_Stop(void)
{ 
    I2C_SDA_O=0;
    I2C_SCL=0; 
  I2C_DELAY();
  I2C_DELAY();
    I2C_SCL=1;
  I2C_DELAY();
  I2C_DELAY();
    I2C_SDA_O=1;
  I2C_DELAY();
  I2C_DELAY();
    I2C_SCL=0;
}

/*
 * I2C_Stop
 * I2CӦ���źţ��ڲ�����
 */
void I2C_Ack(void)
{
  I2C_SCL=0;
  I2C_DELAY();
  
  I2C_SDA_O=0;
  I2C_DELAY();
  
  I2C_SCL=1;
  I2C_DELAY();
  
  I2C_SCL=0;
  I2C_DELAY();
}

/*
 * I2C_NoAck
 * I2C��Ӧ���źţ��ڲ�����
 */
 void I2C_NoAck(void)
{
  I2C_SCL=0;
  I2C_DELAY();
  I2C_SDA_O=1;
  I2C_DELAY();
  I2C_SCL=1;
  I2C_DELAY();
  I2C_SCL=0;
  I2C_DELAY();
}


/*
 * I2C_SendByte
 * I2C�������ݣ��ڲ�����
 */
void I2C_SendByte(uint8 data)
{
  uint8 i=8;
  while(i--)
  {
    I2C_SCL=0;
    if(data&0x80)
      I2C_SDA_O=1; 
    else 
      I2C_SDA_O=0;
    data<<=1;
    I2C_DELAY();
    I2C_SCL=1;
    I2C_DELAY();
  }
  I2C_SCL=0;
  I2C_DELAY();
  I2C_SDA_I=1;
  I2C_DELAY();
  I2C_SCL=1; 
  I2C_DELAY();
  I2C_SCL=0;
}

/*
 * I2C_SendByte
 * I2C�������ݣ��ڲ�����
 */
 uint8 I2C_ReceiveByte(void)  
{ 
  uint8 i=8;
  uint8 ReceiveByte=0;
  
  I2C_SDA_O=1;	
  I2C_SDA_IN();	
  
  while(i--)
  {
    ReceiveByte<<=1;      
    I2C_SCL=0;
    I2C_DELAY();
    I2C_SCL=1;
    I2C_DELAY();	
    if(I2C_SDA_I)
    {
      ReceiveByte|=0x01;
    }

  }
  I2C_SDA_OUT();
  I2C_SCL=0;
  return ReceiveByte;
}

/*
 * I2C_SendByte
 * I2C��ʱ�������ڲ�����
 */
void I2C_Delay(uint16 i)
{	
  while(i) 
    i--;
}