#include "include.h"

extern Z_T z_tangle;
extern A_D AD;
extern Car_Status_e Car_Status;
extern B_R B_Ring;
extern BLOCK Block;
extern motor_P motor_p;
extern uint16 get_add;
uint8 count=0,buzzer_count=0,buzzer_number=0,buzzer_time=0; 
extern uint16 bi,c,d;

uint16 aaa = 0;
extern uint8 b ;


void pitinit(void)
{
  pit_init_ms(PIT0,3); 
  set_vector_handler(PIT0_VECTORn,PIT0_IRQHandler);
  set_irq_priority(PIT0_IRQn,1);
  enable_irq(PIT0_IRQn );
  
  
  pit_init_ms(PIT1,5); 
  set_vector_handler(PIT1_VECTORn,PIT1_IRQHandler);
  set_irq_priority(PIT1_IRQn,2);
  enable_irq(PIT1_IRQn );
}  



void PIT0_IRQHandler(void)
{
  PIT_Flag_Clear(PIT0);//清中断标志位
  ad_scan();
  
  if((gpio_get(PTB22)==1)&&get_add>450&&get_add<650&&Block.b_flag==0&&c>100&&c<250)//可能存在400-480之间没有进中断
  {
    Block.b_flag=1;
    motor_p.ex_speed=400;
  }
  if(Block.b_flag==1)
  {
    block();
  }
  else
  {
    zhi();
    benring();
    direction_out();
  }
 

  count++;
  if(count==7)
  {
    count=0;
    
    speed_contol();
    
    speed_out();
  }

if(B_Ring.ring_inside==1)
  {
      B_Ring.ring_incnt++;
      if(B_Ring.ring_incnt==2500){
        B_Ring.ring_flag=0;
        B_Ring.ring_in=0;
        B_Ring.ring_flag_l=0;
        B_Ring.ring_flag_r=0;
        B_Ring.ring_inside=0;
        B_Ring.ring_incnt=0;
      }
  }


  if(z_tangle.z_k==1)
  {
      z_tangle.z_cnt++;
      if(z_tangle.z_cnt==500){
        z_tangle.z_flag=0;
        z_tangle.z_flag_l=0;
        z_tangle.z_flag_r=0;
        z_tangle.z_cnt=0;
        z_tangle.z_k=0;
      }
    }
  
/*********停车*************/  
  if(b==1)
  {
    aaa++;
    if(aaa==666) 
    {
      aaa = 0;
      b = 2;
    }
  }
  
  
}
void PIT1_IRQHandler(void)
{
  PIT_Flag_Clear(PIT1);
  BIBI();
}

void bibi_init(uint8 t,uint8 l)
{
  buzzer_time=l*5;
  buzzer_number=l*5;
  buzzer_count=t*2;
}

void BIBI()
{
  if(buzzer_count>=1)
    {  
            if(buzzer_number>=1)
            {	
              if(buzzer_count%2==1)   {gpio_set(PTD15,0);}
              else                    {gpio_set(PTD15,1);}
              buzzer_number--;	
            }
            else
            {
              buzzer_number=buzzer_time;
              buzzer_count--;
            }
    }
    else
    {
      gpio_set(PTD15,1);
      buzzer_time=0;
      buzzer_count=0;
    }
}


uint16 get_block()
{
      uint8 high_add=0,low_add=0;
      high_add = i2c_read_reg (I2C1, 0x52, 0x00);
      DELAY_MS(10);
      low_add = i2c_read_reg (I2C1, 0x52, 0x01);
      DELAY_MS(10);
      return (high_add << 8) + low_add;
}

