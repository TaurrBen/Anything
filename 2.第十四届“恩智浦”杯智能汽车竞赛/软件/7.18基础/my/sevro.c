#include "include.h"
extern A_D AD;
extern motor_P motor_p;
Z_T z_tangle ;
B_R B_Ring;
Servo_P Servo_p;
D_P D_p;
BLOCK Block;
Car_Status_e Car_Status = Stop;

float k_d=55;//55
float k_p=12.1;//8.3

/*
k_p     k_d
13      40     600

*/

 void servo_Init(void)
{
    Servo_p.sev_starting = 650;
    FTM_PWM_init(FTM1, FTM_CH0,50, Servo_p.sev_starting);//¶æ»ú
}

void direction_out(void)
{ 
  if((B_Ring.ring_in==1&&B_Ring.ring_inside==0)||(z_tangle.z_flag==1&&z_tangle.z_k==0))
  {
    Servo_p.ser_sum=AD.AD_slanting_left+AD.AD_slanting_right+AD.AD_middle;
    Servo_p.ser_sum=powf(Servo_p.ser_sum,1.46);
    if((B_Ring.ring_flag_l==1)||(z_tangle.z_flag_l==1))
    {
      Servo_p.deviation=1.5*AD.AD_slanting_left-AD.AD_slanting_right;
    }
    else if((B_Ring.ring_flag_r==1)||(z_tangle.z_flag_r==1))
    {
      Servo_p.deviation=AD.AD_slanting_left-1.25*AD.AD_slanting_right;
    }   
  }else{
    Servo_p.ser_sum=AD.AD_left+AD.AD_right+AD.AD_middle;
    Servo_p.ser_sum=powf(Servo_p.ser_sum,1.46);
    Servo_p.deviation=AD.AD_left-AD.AD_right;
  }

   Servo_p.sev_err[0]=449*(Servo_p.deviation-0)/Servo_p.ser_sum;
  
  Servo_p.sev_err[0]=(Servo_p.sev_err[0]*8+Servo_p.sev_err[1]*2)/10;
//  D_p.D_sev_err[0] = Servo_p.sev_err[0];
  Servo_p.sev_ec=Servo_p.sev_err[0]-Servo_p.sev_err[1];

  Servo_p.sev_duty=(k_p*Servo_p.sev_err[0]+k_d*Servo_p.sev_ec);
  Servo_p.sev_err[1]=Servo_p.sev_err[0];
  
  if(Servo_p.sev_duty>200) Servo_p.sev_duty = 200;
  if(Servo_p.sev_duty<-170) Servo_p.sev_duty = -170;
  Servo_p.sev_end=(uint16)(Servo_p.sev_starting-Servo_p.sev_duty);

  FTM_PWM_Duty(FTM1, FTM_CH0, Servo_p.sev_end);

}


void zhi()
{
  if(!((AD.AD_leftangle>90)&&(AD.AD_rightangle>90)))
  {
    if((!z_tangle.z_flag_r)&&(AD.AD_leftangle>90)&&(AD.AD_middle<50)&&(AD.AD_leftangle>AD.AD_rightangle)&&(AD.AD_slanting_left<60)&&(AD.AD_slanting_right<60))
  	{
          z_tangle.z_flag=1;
          z_tangle.z_flag_l=1;
  	}else if((!z_tangle.z_flag_l)&&(AD.AD_rightangle>90)&&(AD.AD_middle<50)&&(AD.AD_rightangle>AD.AD_leftangle)&&(AD.AD_slanting_left<60)&&(AD.AD_slanting_right<60))
  	{
          z_tangle.z_flag_r=1;
          z_tangle.z_flag=1;
  	}
  }
  if(abs(AD.AD_left-AD.AD_right)>15&&z_tangle.z_flag==1)
  {
     z_tangle.z_k=1;
  }
  
}





void benring(void)
{
  if(AD.AD_left+AD.AD_right+AD.AD_middle>190&&AD.AD_middle>60)
  {
      B_Ring.ring_flag=1;
  }
  
  if(B_Ring.ring_flag==1&&(!B_Ring.ring_flag_r)&&AD.AD_slanting_right>110&&AD.AD_slanting_left<AD.AD_slanting_right)
  {
      B_Ring.ring_in=1;
      B_Ring.ring_flag_l=1;
  }else
  if(B_Ring.ring_flag==1&&(!B_Ring.ring_flag_l)&&AD.AD_slanting_left>110&&AD.AD_slanting_left>AD.AD_slanting_right)
  {
      B_Ring.ring_in=1;
      B_Ring.ring_flag_r=1;
  }
  if(B_Ring.ring_in==1&&AD.AD_left<50&&AD.AD_right<50){
    B_Ring.ring_inside=1;
  }
}



/***********************±ÜÕÏ***********************/
void block()
{
  uint8 stage=0;
  
  if(motor_p.get_count<12000) stage=1;
  else if(motor_p.get_count<29000) stage=2;
  else if(motor_p.get_count<33000) stage=3;
  else{
    bibi_init(2,2);
    Block.b_flag=0;
    stage=0;
    motor_p.ex_speed=600;
    motor_p.get_count=0;
  }

  switch(stage)
  {
      case 1:
        FTM_PWM_Duty(FTM1, FTM_CH0, 760);
        break;
      case 2:
        FTM_PWM_Duty(FTM1, FTM_CH0,480);
        break;
      case 3:
        FTM_PWM_Duty(FTM1, FTM_CH0,750);
        break;
  }
}





