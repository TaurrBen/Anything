#include "include.h"
#define BM2 gpio_get(PTC0)
#define BM1 gpio_get(PTC1)
#define BM6 gpio_get(PTB20)
#define BM5 gpio_get(PTB21)
#define BM4 gpio_get(PTB22)
#define BM3 gpio_get(PTB23)

GUI_P GUI_p; 
extern A_D AD;
extern Servo_P Servo_p;
extern B_R B_Ring;
extern motor_P motor_p;
extern Z_T z_tangle;
extern Car_Status_e Car_Status;
extern uint16 a,bi,c,d;
extern uint8 b;
extern float PI_P,PI_I,k_p,k_d;
extern uint16 get_add;
void Oled_Display(void)
{
  LCD_Print(0,0,"P");
  Dis_Num(3,0,GUI_p.page,1);
  LCD_Print(40,0,"L");
  Dis_Num(7,0,GUI_p.line,1);
  LCD_Print(75,0,"U");
  Dis_Num(12,0,GUI_p.unit,1);
  uint8 BM = BM1*1 + BM2*2;
  switch(BM)
  {
    case 0 : GUI_p.unit = 0;break;
    case 1 : GUI_p.unit = 1;break;
    case 2 : GUI_p.unit = 2;break;
    case 3 : GUI_p.unit = 3;break;
    default:break;
  }
  if(GUI_p.page==0)
  {
    
    Dis_Num(0,2,AD.AD_left,3);
    
    Dis_Num(5,2,AD.AD_middle,3);
    
    Dis_Num(10,2,AD.AD_right,3);
    
    Dis_Num(0,4,AD.AD_slanting_left,3);
    
    Dis_Num(5,4,Car_Status,1);
    
    Dis_Num(10,4,AD.AD_slanting_right,3);
    
    Dis_Num(0,6,AD.AD_leftangle,3);
    
    Dis_Num(5,6,get_add,4);
   // Dis_Num(5,6,get_add,4);
//   Dis_Num(5,6,B_Ring.ring_cnt,5);
    
    Dis_Num(10,6,AD.AD_rightangle,3);
    
  }
 if(GUI_p.page==1)
  {
    LCD_Print(10,2,"k_p");
    Dis_Float(6,2,k_p,1);
//    Dis_Float(12,2,fuzzy_kp,1);
    
    
    LCD_Print(10,4,"k_d");
    Dis_Float(6,4,k_d,1);   
//    Dis_Float(12,4,fuzzy_kd,1);
    
    LCD_Print(10,6,"s_duty");
    Dis_Num(8,6,Servo_p.sev_end,3);
    
    LCD_Print(0,2*GUI_p.line,"*");
  }
  else if(GUI_p.page==2)
  {
    LCD_Print(10,2,"ex_s");
    Dis_Float(6,2,motor_p.ex_speed,1);
//    Dis_Float(12,2,fuzzy_speed,1);
   
    LCD_Print(10,4,"PI_P");
    Dis_Float(6,4,PI_P,1);
    LCD_Print(10,6,"PI_I");
    Dis_Float(6,6,PI_I,1);
    LCD_Print(0,2*GUI_p.line,"*");
  }
  else if(GUI_p.page==3)
  {
    Dis_Num(0,2,B_Ring.ring_flag,2);
    Dis_Num(3,2,B_Ring.ring_flag_l,2);
    Dis_Num(6,2,B_Ring.ring_flag_r,2);
    Dis_Num(9,2,B_Ring.ring_in,2);
    Dis_Num(12,2,B_Ring.ring_inside,2);
    Dis_Num(0,4,z_tangle.z_flag,2);
    Dis_Num(3,4,z_tangle.z_flag_l,2);
    Dis_Num(6,4,z_tangle.z_flag_r,2);
    Dis_Num(9,4,z_tangle.z_k,2);
    Dis_Num(0,6,BM1*1+BM2*2+BM3*3+BM4*4+BM5*5+BM6*6,2);
    Dis_Num(3,6,bi,3);
    Dis_Num(7,6,c,3);
    Dis_Num(11,6,d,3);
  }
  
}
uint8 key(void)
{
    uint8 n=0;
    if(gpio_get(PTA16)==0)//翻页+
    {
        n=1;
        //systick_delay_us(100);
        while(gpio_get(PTA16)==0);
    }

    else if(gpio_get(PTA12)==0)//数值-
    {
        n=2;
        while(gpio_get(PTA12)==0);
    }

    else if(gpio_get(PTA15)==0)//换行
    {
      n=3;
      while(gpio_get(PTA15)==0);
    }
    else if(gpio_get(PTA14)==0)//数值+
    {
      n=4;
      while(gpio_get(PTA14)==0);
    }
    else if(gpio_get(PTA13)==0)//翻页-
    {
      n=5;
      while(gpio_get(PTA13)==0);
    }
    return n;
}

void KeyScan(void)
{
    uint8 m=key( );
    switch(m)
    {
        case 1:
        {
            LCD_Fill(0x00);
            GUI_p.page+=1;
            if(GUI_p.page>3) GUI_p.page=0;
            break;
         }
        case 2:
        {
          if(GUI_p.page==1&&GUI_p.line==1)
          {
              switch(GUI_p.unit)
              {
                case 0:k_p -=0.1;break;
                case 1:k_p -=0.5;break;
                case 2:k_p -=1;break;
                case 3:k_p -=5;break;
                default:break;
              }  
          }
          else if(GUI_p.page==1 && GUI_p.line==2)
          {
              switch(GUI_p.unit)
              {
                case 0:k_d -=1;break;
                case 1:k_d -=5;break;
                case 2:k_d -=10;break;
                case 3:k_d -=50;break;
                default:break;
              }
          }
          else if(GUI_p.page==2 && GUI_p.line==1)
          {
              switch(GUI_p.unit)
              {
                case 0:motor_p.ex_speed -=10;break;
                case 1:motor_p.ex_speed -=50;break;
                case 2:motor_p.ex_speed -=100;break;
                case 3:motor_p.ex_speed -=200;break;
                default:break;
              }
          }
          else if(GUI_p.page==2 && GUI_p.line==2)
          {
              switch(GUI_p.unit)
              {
                case 0:PI_P -=0.1;break;
                case 1:PI_P -=0.5;break;
                case 2:PI_P -=1;break;
                case 3:PI_P -=5;break;
                default:break;
              }
          }
          else if(GUI_p.page==2 && GUI_p.line==3)
          {      
              switch(GUI_p.unit)
              {
                case 0:PI_I -=0.1;break;
                case 1:PI_I -=0.5;break;
                case 2:PI_I -=1;break;
                case 3:PI_I -=5;break;
                default:break;
              }
          }
          break;
        }
        case 3:
        {
            if(GUI_p.page==0)
            {
              b=0;
              if(Car_Status ==Stop) {Car_Status = Run;break;}
              if(Car_Status ==Run) {Car_Status = Stop;break;}
            }
            if(GUI_p.page==1)
            {
                GUI_p.line+=1;
                LCD_Fill(0x00);
                if(GUI_p.line>2) GUI_p.line=1;
                break;
            }
             if(GUI_p.page==2)
            {
                GUI_p.line+=1;
                LCD_Fill(0x00);
                if(GUI_p.line>3) GUI_p.line=1;
                break;
            }
            break;
        }
        case 4:
        {
          if(GUI_p.page==1&&GUI_p.line==1)
          {
              switch(GUI_p.unit)
              {
                case 0:k_p +=0.1;break;
                case 1:k_p +=0.5;break;
                case 2:k_p +=1;break;
                case 3:k_p +=5;break;
                default:break;
              }  
          }
          else if(GUI_p.page==1 && GUI_p.line==2)
          {
              switch(GUI_p.unit)
              {
                case 0:k_d +=1;break;
                case 1:k_d +=5;break;
                case 2:k_d +=10;break;
                case 3:k_d +=50;break;
                default:break;
              }
          }
          else if(GUI_p.page==2 && GUI_p.line==1)
          {
              switch(GUI_p.unit)
              {
                case 0:motor_p.ex_speed +=10;break;
                case 1:motor_p.ex_speed +=50;break;
                case 2:motor_p.ex_speed +=100;break;
                case 3:motor_p.ex_speed +=200;break;
                default:break;
              }
          }
          else if(GUI_p.page==2 && GUI_p.line==2)
          {
              switch(GUI_p.unit)
              {
                case 0:PI_P +=0.1;break;
                case 1:PI_P +=0.5;break;
                case 2:PI_P +=1;break;
                case 3:PI_P +=5;break;
                default:break;
              }
          }
          else if(GUI_p.page==2 && GUI_p.line==3)
          {      
              switch(GUI_p.unit)
              {
                case 0:PI_I +=0.1;break;
                case 1:PI_I +=0.5;break;
                case 2:PI_I +=1;break;
                case 3:PI_I +=5;break;
                default:break;
              }
          }
          break;
        }
        case 5:
        {
            LCD_Fill(0x00);
            GUI_p.page-=1;
            if(GUI_p.page<0)
              GUI_p.page=3;
            break;
         }
        default:break;  
    }
}




