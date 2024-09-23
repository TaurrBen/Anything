#ifndef _MOTOR_H
#define _MOTOR_H


typedef struct node
{
    float get_speed;
    float speed_err;
    float speed_err_last;
    float speed;
    float ex_speed;
    uint16 get_count;
}motor_P;

void motor_Init();
void speed_get();
void speed_contol();
void speed_out();

#endif
