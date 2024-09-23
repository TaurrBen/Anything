#include  "include.h"

#define Key_Up gpio_get(PTC8)  //up   
#define Key_Dn gpio_get(PTC13) //down 
#define Key_Lt gpio_get(PTC10) //left
#define Key_Rt gpio_get(PTC19) //right
#define Key_Md gpio_get(PTC6)  //middle

#define Boma6 gpio_get(PTD1)   //挡位选择
#define Boma5 gpio_get(PTD4)  
#define Boma4 gpio_get(PTD6)
#define Boma3 gpio_get(PTD9)     
#define Boma2 gpio_get(PTD11)  //选择模式 00 无        01   Flash
#define Boma1 gpio_get(PTD13)  //         10 Car_Run   11   Normalization

extern uint16 AD_right,AD_middle,AD_straight_right,AD_straight_left,AD_left,k_d;
extern uint16 sum,sev_end,G;
extern float ex_speed,PI_P,PI_I,k_p;
extern uint32 speed_duty;
extern uint8 ring_flag;
extern uint8 ring_in;
extern uint8 ring_out,ring_flag_l, ring_flag_r,ring_inside;
extern uint8 z_flag;

Menu_e Menu = Other;
uint8 Page = 1,Line = 1,Unit = 1;
void OLED_Display(void)
{
  if(Page==1)
  {
    Dis_Num(0,0,AD_left,3);
    Dis_Num(5,0,AD_middle,3);
    Dis_Num(10,0,AD_right,3);
    Dis_Num(0,2,AD_straight_left,3);
    Dis_Num(10,2,AD_straight_right,3);
    
    
  }
  else if(Page==2)
  {
    LCD_Print(0,0,"k_p");
    Dis_Float(4,0,k_p,1);
    LCD_Print(72,4,"k_d");
    Dis_Num(13,4,k_d,1);
    
    LCD_Print(80,6,"sev_duty");
    Dis_Num(0,6,sev_end,3);
  }
  else if(Page==3)
  {
    Dis_Float(6,0,ex_speed,3);
    Dis_Float(6,2,PI_P,3);
    Dis_Float(6,4,PI_I,3);
  }
  else if(Page==4)
  {
    Dis_Num(0,0,ring_flag,2);
    Dis_Num(3,0,ring_flag_l,2);
    Dis_Num(6,0,ring_flag_r,2);
    Dis_Num(0,4,ring_in,2);
    Dis_Num(9,4,ring_inside,2);
    Dis_Num(6,6,z_flag,2);
  }
}
void TFT18_Display(void)
{
    
}
uint8 Key_Scan(void)
{
    if(Key_Up == 0)
    {
        DELAY_MS(5);
		if (Key_Up == 0)
		{
			while (!Key_Up);
			return 8;
		}
    }
    if(Key_Dn == 0)
    {
        DELAY_MS(5);
		if (Key_Dn == 0)
		{
			while (!Key_Dn);
			return 2;
		}
    }
    if(Key_Lt == 0)
    {
        DELAY_MS(5);
		if (Key_Lt == 0)
		{
			while (!Key_Lt);
			return 5;
		}
    }
    if(Key_Rt == 0)
    {
        DELAY_MS(5);
		if (Key_Rt == 0)
		{
			while (!Key_Rt);
			return 4;
		}
    }
    if(Key_Md == 0)
    {
        DELAY_MS(5);
		if (Key_Md == 0)
		{
			while (!Key_Md);
			return 6;
		}
    }
    return 0;
}
void Clear_Screen(void) 
{
	static  uint8 Last_Menu = 4;
	
	if(Last_Menu != Menu)
	{
		LCD_Fill(0x00);
		Page = 1;
		Line = 1;
		Last_Menu = Menu;
	}
}
void Menu_Define(void) 
{
    uint8 mode = 0;
	mode = Boma3*1 ;//+ Boma2*2;
	switch(mode)
	{
        case 0:
        {
            Menu=Other;
            Clear_Screen();
            break;  
        }
	case 1:
	{
	Menu=Car_Run;
	Clear_Screen();
	break;
	}

  }
}
void Boma_Oled(void) 
{
	Menu_Define();
	Clear_Screen();
    if(Menu == Other)
    {
        Other_Oled();
        return;
    }
	
    if(Menu == Car_Run)
	{
		OLED_Display();
		return;
	}
}
//显示的东西
void Other_Oled(void)
{
    LCD_Print(5,3,"HELLO,MAFEI");
}


void Key_Control(void)
{

    if(Menu == Car_Run)     Car_Run_Key();
    if(Boma5 == 0 && Boma6 == 1)  Unit = 1;
    else if(Boma5 ==1 && Boma6 == 0) Unit = 2;
    else if(Boma5 ==1 && Boma6 ==1) Unit = 3;
    else Unit=4;
}
void Car_Run_Key(void)
{
    switch(Key_Scan())
	{
		case 4:  //加页
		{
			Page += 1;
			LCD_Fill(0x00);
			if(Page == 6) Page = 1;
			break;
		}
		case 5:  //减页
		{
			Page -= 1;
			LCD_Fill(0x00);
			if(Page == 0) Page = 5;
			break;
		}
        case 6:  //换行
       {
           Line ++;
           LCD_Fill(0x00);
          if(Line == 4) Line=1;
           break;
       }
		case 8:  //加
		{
                  if(Page==2)
                  {
                     if(Line == 1)
                     {
					switch(Unit)
					{
						case 1:{k_p += 1;break;}
						case 2:{k_p += 3;break;}        
						case 3:{k_p += 5;break;}
                                                case 4:{k_p += 7;break;}
					}
//					Flash_Updata();
					break;
				}
                           if(Line == 2)
                     {
					switch(Unit)
					{
						case 1:{k_d += 0.01;break;}
						case 2:{k_d += 0.1;break;}        
						case 3:{k_d += 0.5;break;}
                                                case 4:{k_d += 1;break;}
					}
//					Flash_Updata();
					break;
				}
                     
                  
                  }
                  if(Page==3)
                  {
                     if(Line == 1)
                     {
					switch(Unit)
					{
						case 1:{ex_speed += 10;break;}
						case 2:{ex_speed += 30;break;}        
						case 3:{ex_speed += 50;break;}
                                                case 4:{ex_speed += 70;break;}
					}
//					Flash_Updata();
					break;
				}
                           if(Line == 2)
                     {
					switch(Unit)
					{
						case 1:{PI_P += 0.01;break;}
						case 2:{PI_P += 0.1;break;}        
						case 3:{PI_P += 0.5;break;}
                                                case 4:{PI_P += 1;break;}
					}
//					Flash_Updata();
					break;
				}
                      if(Line == 3)
                     {
					switch(Unit)
					{
						case 1:{PI_I += 0.01;break;}
						case 2:{PI_I += 0.1;break;}        
						case 3:{PI_I += 0.5;break;}
                                                case 4:{PI_I += 1;break;}
					}
//					Flash_Updata();
					break;
				}
                     
                  
                  }
                }
                  case 2:  //加
		{
                  if(Page==2)
                  {
                     if(Line == 1)
                     {
					switch(Unit)
					{
						case 1:{k_p -= 1;break;}
						case 2:{k_p -= 3;break;}        
						case 3:{k_p -= 5;break;}
                                                case 4:{k_p -= 7;break;}
					}
//					Flash_Updata();
					break;
				}
                           if(Line == 2)
                     {
					switch(Unit)
					{
						case 1:{k_d -= 0.01;break;}
						case 2:{k_d -= 0.1;break;}        
						case 3:{k_d -= 0.5;break;}
                                                case 4:{k_d -= 1;break;}
					}
//					Flash_Updata();
					break;
				}
                     
                  
                  }
                  if(Page==3)
                  {
                     if(Line == 1)
                     {
					switch(Unit)
					{
						case 1:{ex_speed -= 10;break;}
						case 2:{ex_speed -= 30;break;}        
						case 3:{ex_speed -= 50;break;}
                                                case 4:{ex_speed -= 70;break;}
					}
//					Flash_Updata();
					break;
				}
                           if(Line == 2)
                     {
					switch(Unit)
					{
						case 1:{PI_P -= 0.01;break;}
						case 2:{PI_P -= 0.1;break;}        
						case 3:{PI_P -= 0.5;break;}
                                                case 4:{PI_P -= 1;break;}
					}
//					Flash_Updata();
					break;
				}
                      if(Line == 3)
                     {
					switch(Unit)
					{
						case 1:{PI_I -= 0.01;break;}
						case 2:{PI_I -= 0.1;break;}        
						case 3:{PI_I -= 0.5;break;}
                                                case 4:{PI_I -= 1;break;}
					}
//					Flash_Updata();
					break;
				}
                     
                  
                  }
 
        }
                
    default:
        break;
	}
        
}

