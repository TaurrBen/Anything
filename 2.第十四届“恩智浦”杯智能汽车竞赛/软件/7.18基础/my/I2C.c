#include "include.h"
#define USERI2C_SDA_IN()  {gpio_ddr (PTB1, GPI); }
#define USERI2C_SDA_OUT() {gpio_ddr (PTB1, GPO); }
void IIC_Init(void)
{
  
gpio_init (PTE12, GPO,1);
gpio_init (PTE11, GPO,1);
//USERI2C_SCL=1;
//USERI2C_SDA=1;

}