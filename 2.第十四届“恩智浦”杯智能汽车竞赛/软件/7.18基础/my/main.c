#include "include.h"

extern A_D AD;
extern Servo_P Servo_p;
extern Z_T z_tangle;
extern B_R B_Ring;
extern motor_P motor_p;
extern GUI_P GUI_p;
extern D_P D_p;
extern Car_Status_e Car_Status;
extern BLOCK Block;

uint16 Debug_p[8];
uint16 a=0,bi=0,c=0,d=0;
uint8 b=0;
uint16 get_add=0;
void Init(void)
{
    DisableInterrupts;
    uart_init (UART3, 115200);
    servo_Init();
    ad_init( );
    LCD_Init( ); 
    motor_Init( );
    pitinit();
    i2c_init(I2C1,9600);
    My_IIC_Port_Init();
    /************GUI********************/
    GUI_p.page=0;
    GUI_p.line=1;
    GUI_p.unit=0;
    /***********·äÃùÆ÷***************/
    gpio_init(PTD15,GPO,1);
    gpio_init(PTE10,GPI,1);
    gpio_init(PTE9,GPI,1);
    gpio_init(PTE8,GPI,1);
    port_init (PTD15,  ALT1|ODO|PULLUP );
    port_init (PTE10,  ALT1|ODO|PULLUP );
    port_init (PTE9,  ALT1|ODO|PULLUP );
    port_init (PTE8,  ALT1|ODO|PULLUP );
    /***********Ò£¿Ø***************/
    gpio_init(PTA26,GPI,1);//B
    gpio_init(PTA27,GPI,1);//D
    gpio_init(PTA28,GPI,1);//A
    gpio_init(PTA29,GPI,1);//C
    /************ÎåÖá************/
    gpio_init (PTA12,GPI,1);//ÓÒ
    port_init(PTA12,  ALT1 | PULLUP);
     
    gpio_init (PTA13,GPI,1);//ÉÏ
    port_init(PTA13,  ALT1 | PULLUP);
      
    gpio_init (PTA14,GPI,1);//×ó
    port_init(PTA14,  ALT1 | PULLUP);
      
    gpio_init (PTA15,GPI,1);//ÖÐ
    port_init(PTA15,  ALT1 | PULLUP);
      
    gpio_init (PTA16,GPI,1);//ÏÂ
    port_init(PTA16,  ALT1 | PULLUP);
    
    /**********²¦Âë**********/
    gpio_init(PTC0,GPI,1);
    port_init(PTC0,  ALT1 | PULLUP);//1
    
    gpio_init(PTC1,GPI,1);
    port_init(PTC1,  ALT1 | PULLUP);//2
    
    gpio_init(PTB23,GPI,1);
    port_init(PTB23, ALT1 | PULLUP);//3
    
    gpio_init(PTB22,GPI,1);
    port_init(PTB22, ALT1 | PULLUP);//boma4
    
    gpio_init(PTB21,GPI,1);
    port_init(PTB21, ALT1 | PULLUP);//boma5
    
    gpio_init(PTB20,GPI,1);
    port_init(PTB20, ALT1 | PULLUP);//boma6
    /***********************/   
    gpio_init (PTE26,GPO,1);
    gpio_init (PTA17,GPO,1);
    /***************/
    i2c_write_reg(I2C1, 0x52, 0x09, 0x01); 
    DELAY_MS(30);
    i2c_write_reg(I2C1, 0x52, 0x08, 0x00);
    DELAY_MS(30);
    i2c_write_reg(I2C1, 0x52, 0x07, 0x00);
    DELAY_MS(30);
    i2c_write_reg(I2C1, 0x52, 0x06, 0x00);
    DELAY_MS(30);
    
    EnableInterrupts;
}

void main()
{    
    Init();
    while(1)
    {
      
      if(gpio_get(PTB23)==1)
      {
         if((gpio_get(PTE10)==0)||(gpio_get(PTE9)==0)||(gpio_get(PTE8)==0))
        {
          if(b == 2) 
          {
             b = 0;
             Car_Status = Stop;
          } 
          else b=1;
        }
      }
//       Debug_p[0] = B_Ring.ring_flag;
//       Debug_p[1] = B_Ring.ring_flag_l;
//       Debug_p[2] = B_Ring.ring_flag_r;
//       Debug_p[3] = B_Ring.ring_in;
//       Debug_p[4] = B_Ring.ring_inside;
//       Debug_p[5] = 0;
//       Debug_p[6] = 0;
//       Debug_p[7] = 0; 
      

        
       Debug_p[0] = AD.AD_left;
       Debug_p[1] = AD.AD_leftangle;
       Debug_p[2] = AD.AD_slanting_left;
       Debug_p[3] = AD.AD_middle;
       Debug_p[4] = AD.AD_slanting_right;
       Debug_p[5] = AD.AD_rightangle;
       Debug_p[6] = AD.AD_right;
       Debug_p[7] = AD.AD_left+AD.AD_right+AD.AD_middle;
              
//       Debug_p[0] = D_p.D_deviation[0];
//       Debug_p[1] = D_p.D_deviation[1];
//       Debug_p[2] = D_p.D_sev_err[0];
//       Debug_p[3] = D_p.D_sev_err[1];
//       Debug_p[4] = D_p.D_deviation[0] - D_p.D_deviation[1];
//       Debug_p[5] = D_p.D_sev_err[0] - D_p.D_sev_err[1];
//       Debug_p[6] = 0;
//       Debug_p[7] = 0;
        
//       Debug_p[0] = motor_p.get_speed;
//       Debug_p[1] = motor_p.ex_speed;
//       Debug_p[2] = motor_p.speed;
//       Debug_p[3] = Servo_p.sev_starting;
//       Debug_p[4] = Servo_p.sev_end;
//       Debug_p[5] = 0;
//       Debug_p[6] = 0;
//       Debug_p[7] = 0;  
      
 
       get_add=get_block();
       GYVL53L0X_Read_data( 0x29, &a, &bi, &c, &d);
//       Debug_p[0] = a;
//       Debug_p[1] = bi;
//       Debug_p[2] = c;
//       Debug_p[3] = d;
//       Debug_p[4] = 0;
//       Debug_p[5] = 0;
//       Debug_p[6] = 0;
//       Debug_p[7] = 0;
       Oled_Display();
       KeyScan();
       vcan_sendware(Debug_p, sizeof(Debug_p));
       
    }        
}
