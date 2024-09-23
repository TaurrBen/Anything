#ifndef _AD_H
#define _AD_H


typedef struct A
{
  uint8 AD_left;
  uint8 AD_slanting_left;
  uint8 AD_leftangle;
  uint8 AD_middle;
  uint8 AD_rightangle;
  uint8 AD_slanting_right;
  uint8 AD_right;
  uint8 Var_Max[7];
  uint8 Var_Min[7];
  uint8 Voltage;
}A_D;


void ad_init(void);
uint8 Bubble(uint8 *v); 
void ad_scan();
double fabss(double a);

#endif


/****
直电感 150  //
普通左右 90//30
中间电感  110//60
斜电感  90//60
****/
