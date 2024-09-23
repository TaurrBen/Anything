#ifndef _pit_h
#define _pit_h

#include "include.h"

void pitinit(void);
void PIT0_IRQHandler(void);
void PIT1_IRQHandler(void);
void BIBI(void);
void bibi_init(uint8 t,uint8 l);
uint16 get_block(void);

#endif
