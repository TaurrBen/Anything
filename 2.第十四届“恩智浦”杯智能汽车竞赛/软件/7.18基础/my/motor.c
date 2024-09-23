#include "include.h"
extern B_R B_Ring;
extern BLOCK Block;
extern Car_Status_e Car_Status;
float PI_P=10.9;
float PI_I=11.1;
motor_P motor_p;
//70/84  300//100  28cm/s

/*V      P       I hh
 700     10.9    11.1
 600     12.1    9.3     
 800     12.3    11.8                             
*/
void motor_Init()
{
    motor_p.ex_speed=600;
    FTM_PWM_init(FTM0, FTM_CH1,8000,0);//µç»ú
    FTM_PWM_init(FTM0, FTM_CH2,8000,0);
    gpio_init (PTD12, GPI,1);
    port_init (PTC5, ALT4||PULLUP);
    lptmr_pulse_init(LPT0_ALT2,0xFFFF,LPT_Rising);
}

void speed_contol()
{
  motor_p.get_speed=lptmr_pulse_get( ); 
  lptmr_pulse_clean(); 
  if(Block.b_flag==1){
    motor_p.get_count+=motor_p.get_speed;
  }
  
  motor_p.speed_err=motor_p.ex_speed-motor_p.get_speed;
  motor_p.speed+=(PI_P*(motor_p.speed_err-motor_p.speed_err_last)+PI_I*motor_p.speed_err);
  motor_p.speed_err_last=motor_p.speed_err;
  
  if(motor_p.speed>9999)
  motor_p.speed=9999;
  else if(motor_p.speed<=0)
  motor_p.speed=0;
}

void speed_out()
{
  if(Car_Status==Run)
  {
    FTM_PWM_Duty(FTM0, FTM_CH1,0);
    FTM_PWM_Duty(FTM0, FTM_CH2,(uint16)(motor_p.speed));
  }
  else
  {
    FTM_PWM_Duty(FTM0, FTM_CH1,0);
    FTM_PWM_Duty(FTM0, FTM_CH2,0);
  }
}
