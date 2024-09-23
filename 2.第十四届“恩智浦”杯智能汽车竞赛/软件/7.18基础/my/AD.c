#include "include.h"
A_D AD;

void ad_init( void)
{
   
    adc_init(ADC0_SE9);//PTB1  AD1   right

    adc_init(ADC0_SE12);//PTB2  AD2   left

    adc_init(ADC0_SE13);//PTB3  AD3   midlle

    adc_init(ADC1_SE10);//PTB4  AD4   

    adc_init(ADC1_SE11);//PTB5  AD5   ÓÒÐ±

    adc_init(ADC1_SE12);//PTB6  AD6   ×óÐ±

    adc_init(ADC1_SE13);//PTB7  AD7  ÓÒÖ±

    adc_init(ADC1_SE14);//PTB10  AD9  ×óÖ±
    

}
double fabss(double a)
{
  
  double b = a > 0.0?a:-a;
  
  return b;
  
}
int intabs(int a)
{
  
  int b=a>0?a:-a;
  
  return b;
  
}

uint8 choose(uint8 *a,int n)
{  
    int i,index,flag,j,count=0;	 
    for(j=0;j<n-1;j++){  
	index=j;
	for(i=j+1;i<n;i++){
	    count++;
	    if(a[index]>a[i]){
	       index=i;
	    }
	 }
	if(j==10) break;
	flag=a[j];
	a[j]=a[index];
	a[index]=flag;}
    return (a[8]+a[9]+a[10])/3;
}


void ad_scan(void)
{

  uint8 var_left[20]={0};
  
  uint8 var_leftangle[20]={0};//×óÖ±
    
  uint8 var_slanting_left[20]={0};//×óÐ±
  
  uint8 var_middle[20]={0};
  
  uint8 var_slanting_right[20]={0};
  
  uint8 var_rightangle[20]={0};
  
  uint8 var_right[20]={0};
  
  for(int i=0;i<20;i++)
    var_right[i]= adc_once(ADC0_SE9,ADC_8bit);
  
  for(int i=0;i<20;i++)
    var_left[i]=adc_once(ADC0_SE12, ADC_8bit);
  
  for(int i=0;i<20;i++)
    var_middle[i]= adc_once(ADC0_SE13, ADC_8bit);  
  
  for(int i=0;i<20;i++)
    var_slanting_right[i]= adc_once(ADC1_SE11, ADC_8bit); 
  
  for(int i=0;i<20;i++)
    var_slanting_left[i]=adc_once(ADC1_SE12, ADC_8bit);
 
  for(int i=0;i<20;i++)
    var_rightangle[i]= adc_once(ADC1_SE13, ADC_8bit); 
  
  for(int i=0;i<20;i++)
    var_leftangle[i]=adc_once(ADC1_SE14, ADC_8bit);

  
  AD.AD_left=choose(var_left,20);
  
  AD.AD_leftangle=choose(var_leftangle,20);
  
  AD.AD_slanting_left=choose(var_slanting_left,20);
  
  AD.AD_middle=choose(var_middle,20);
  
  AD.AD_slanting_right=choose(var_slanting_right,20);
  
  AD.AD_rightangle=choose(var_rightangle,20);

  AD.AD_right=choose(var_right,20);
  
  
//  if(AD.Var_Max[0] < AD.AD_left)                 AD.Var_Max[0] = AD.AD_left;
//  if(AD.Var_Min[0] > AD.AD_left)                 AD.Var_Min[0] = AD.AD_left;
//  
//  if(AD.Var_Max[1] < AD.AD_leftangle)            AD.Var_Max[1] = AD.AD_leftangle;
//  if(AD.Var_Min[1] > AD.AD_leftangle)            AD.Var_Min[1] = AD.AD_leftangle;
//  
//  if(AD.Var_Max[2] < AD.AD_slanting_left)        AD.Var_Max[2] = AD.AD_slanting_left;
//  if(AD.Var_Min[2] > AD.AD_slanting_left)        AD.Var_Min[2] = AD.AD_slanting_left;
//  
//  if(AD.Var_Max[3] < AD.AD_middle)               AD.Var_Max[3] = AD.AD_middle;
//  if(AD.Var_Min[3] > AD.AD_middle)               AD.Var_Min[3] = AD.AD_middle;
//  
//  if(AD.Var_Max[4] < AD.AD_slanting_right)       AD.Var_Max[4] = AD.AD_slanting_right;
//  if(AD.Var_Min[4] > AD.AD_slanting_right)       AD.Var_Min[4] = AD.AD_slanting_right;
//  
//  if(AD.Var_Max[5] < AD.AD_rightangle)           AD.Var_Max[5] = AD.AD_rightangle;
//  if(AD.Var_Min[5] > AD.AD_rightangle)           AD.Var_Min[5] = AD.AD_rightangle;
//  
//  if(AD.Var_Max[6] < AD.AD_right)                AD.Var_Max[6] = AD.AD_right;
//  if(AD.Var_Min[6] > AD.AD_right)                AD.Var_Min[6] = AD.AD_right;
//  
//  
//  if(AD.Var_Min[0] >= AD.AD_left)   AD.AD_left = 1;
//  else AD.AD_left =  (uint8)(100*(AD.AD_left - AD.Var_Min[0])/AD.Var_Max[0] - AD.Var_Min[0]);
//  
//  if(AD.Var_Min[1] >= AD.AD_leftangle)   AD.AD_leftangle = 1;
//  else AD.AD_leftangle =  (uint8)(200*(AD.AD_leftangle - AD.Var_Min[1])/AD.Var_Max[1] - AD.Var_Min[1]);
//  
//  if(AD.Var_Min[2] >= AD.AD_slanting_left)   AD.AD_slanting_left = 1;
//  else AD.AD_slanting_left =  (uint8)(200*(AD.AD_slanting_left - AD.Var_Min[2])/AD.Var_Max[2] - AD.Var_Min[2]);
//  
//  if(AD.Var_Min[3] >= AD.AD_middle)   AD.AD_middle = 1;
//  else AD.AD_middle =  (uint8)(100*(AD.AD_middle - AD.Var_Min[3])/AD.Var_Max[3] - AD.Var_Min[3]);
//  
//  if(AD.Var_Min[4] >= AD.AD_slanting_right)   AD.AD_slanting_right = 1;
//  else AD.AD_slanting_right =  (uint8)(200*(AD.AD_slanting_right - AD.Var_Min[4])/AD.Var_Max[4] - AD.Var_Min[4]);
//  
//  if(AD.Var_Min[5] >= AD.AD_rightangle)   AD.AD_rightangle = 1;
//  else AD.AD_rightangle =  (uint8)(200*(AD.AD_rightangle - AD.Var_Min[5])/AD.Var_Max[5] - AD.Var_Min[5]);
//  
//  if(AD.Var_Min[6] >= AD.AD_right)   AD.AD_right = 1;
//  else AD.AD_right =  (uint8)(100*(AD.AD_right - AD.Var_Min[6])/AD.Var_Max[6] - AD.Var_Min[6]);
  
  
  
  if(AD.Var_Min[0] >= AD.AD_left)   AD.AD_left = 1;
  else AD.AD_left =  (uint8)(100*(AD.AD_left - AD.Var_Min[0])/255 - AD.Var_Min[0]);
  AD.AD_left = AD.AD_left > 0 ? AD.AD_left:1;
  
  if(AD.Var_Min[1] >= AD.AD_leftangle)   AD.AD_leftangle = 1;
  else AD.AD_leftangle =  (uint8)(200*(AD.AD_leftangle - AD.Var_Min[1])/255 - AD.Var_Min[1]);
  AD.AD_leftangle = AD.AD_leftangle > 0 ? AD.AD_leftangle:1;
  
  if(AD.Var_Min[2] >= AD.AD_slanting_left)   AD.AD_slanting_left = 1;
  else AD.AD_slanting_left =  (uint8)(200*(AD.AD_slanting_left - AD.Var_Min[2])/255 - AD.Var_Min[2]);
  AD.AD_slanting_left = AD.AD_slanting_left > 0 ? AD.AD_slanting_left:1;
  
  if(AD.Var_Min[3] >= AD.AD_middle)   AD.AD_middle = 1;
  else AD.AD_middle =  (uint8)(100*(AD.AD_middle - AD.Var_Min[3])/207 - AD.Var_Min[3]);
  AD.AD_middle = AD.AD_middle > 0 ? AD.AD_middle:1;
  
  if(AD.Var_Min[4] >= AD.AD_slanting_right)   AD.AD_slanting_right = 1;
  else AD.AD_slanting_right =  (uint8)(200*(AD.AD_slanting_right - AD.Var_Min[4])/255 - AD.Var_Min[4]);
  AD.AD_slanting_right = AD.AD_slanting_right > 0 ? AD.AD_slanting_right:1;
  
  if(AD.Var_Min[5] >= AD.AD_rightangle)   AD.AD_rightangle = 1;
  else AD.AD_rightangle =  (uint8)(200*(AD.AD_rightangle - AD.Var_Min[5])/255 - AD.Var_Min[5]);
  AD.AD_rightangle = AD.AD_rightangle > 0 ? AD.AD_rightangle:1;
  
  if(AD.Var_Min[6] >= AD.AD_right)   AD.AD_right = 1;
  else AD.AD_right =  (uint8)(100*(AD.AD_right - AD.Var_Min[6])/255 - AD.Var_Min[6]);
  AD.AD_right = AD.AD_right > 0 ? AD.AD_right:1;
}



