#ifndef _SERVO_H
#define _SERVO_H

typedef struct Z
{
  uint8 z_flag;
  uint8 z_flag_l;
  uint8 z_flag_r;
  uint16 z_cnt;
  uint8 z_k;
}Z_T;

typedef struct R
{
  uint8 ring_flag;
  uint8 ring_flag_l;
  uint8 ring_flag_r;
  uint8 ring_in;
  uint8 ring_inside;
  uint16 ring_incnt;
}B_R;

typedef struct B
{
  uint8 b_flag;
}BLOCK;

typedef struct S
{
    uint16 sev_starting;
    uint16 sev_end;     //720--520--350  ср 600
    double ser_sum;
    double sev_duty;
    double deviation;
    double sev_err[2];
    double sev_ec;
}Servo_P;

typedef struct D
{
    double D_deviation[2];
    double D_sev_err[2];
}D_P;

typedef enum 
{
  Run =0,
  Stop =1,
  Stay =2,
} Car_Status_e;

void servo_Init(void);
void direction_out(void);
void zhi(void);
void benring(void);
void block(void);
#endif
