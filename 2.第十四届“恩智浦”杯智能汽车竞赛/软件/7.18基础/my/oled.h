#ifndef __OLED_H__
#define __OLED_H__ 

#include "include.h"

typedef struct G
{
    uint8 line;
    uint8 unit;
    int8 page;
}GUI_P;

void Oled_Display(void);
void KeyScan(void);
uint8 key(void);


#endif
