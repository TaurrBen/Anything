//fft.c 基-2按时间抽取FFT
#include <stdio.h>
#define _USE_MATH_DEFINES//使用math.h里面的π常量
#include <math.h>
#include <malloc.h>
#define N 65536//最大长度
#define MIN 0.00000000001//小于MIN的值被视为0
typedef struct {       //定义一个结构体表示复数的类型
	double re;//实部
	double im;//虚部
}complex;
complex x[N], *W;   //x是输入序列，W是旋转因子
FILE *fp,*fp2;

int size = 0;   //定义数据长度

void output()//输出复数
{
	for (int i = 0; i < size; i++)
	{
		printf("%.6f", x[i].re);//输出复数的实部
		if (x[i].im >= MIN)
			printf("+%.6fj\n", x[i].im);//当复数的虚部大于MIN时，输出+ 虚部 j的形式
		else if (fabs(x[i].im) < MIN)
			printf("\n");//当虚部小于MIN时，跳过虚部，不输出
		else
			printf("%.6fj\n", x[i].im);//上述两个条件除外的形式，输出 虚部 j的形式
	}
}

void reverse()//输入序列的整序
{
	complex temp;
	int i = 0, j = 0, k = 0;
	double t;
	for (i = 0; i < size; i++)
	{
		k = i;
		j = 0;
		t = (log(size) / log(2));//算出序列的级数
		while ((t--) > 0)//利用按位与以及循环实现码位颠倒
		{
			j = j << 1;
			j |= (k & 1);
			k = k >> 1;
		}
		if (j > i) //将x(n)的码位互换
		{
			temp = x[i];
			x[i] = x[j];
			x[j] = temp;
		}
	}
	output();//输出整序后的序列
}
void twiddleFactor()//旋转因子
{
	W = (complex*)malloc(sizeof(complex) * size);//给指针分配size的空间 size是数据长度
	for (int i = 0; i < size; i++)
	{
		W[i].re = cos(2 * M_PI / size * i);//使用欧拉公式表示
		W[i].im = -1 * sin(2 * M_PI / size * i);
	}
}
void add(complex a, complex b, complex* c)//复数加法
{
	c->re = a.re + b.re;
	c->im = a.im + b.im;
}
void subtract(complex a, complex b, complex* c)//复数加法
{
	c->re = a.re - b.re;
	c->im = a.im - b.im;
}
void multiply(complex a, complex b, complex* c)//复数乘法
{
	c->re = a.re * b.re - a.im * b.im;
	c->im = a.re * b.im + a.im * b.re;
}
void addZero()//输入序列补零
{
	int nextpowto2 = 1;
	while (nextpowto2 < size)//找到大于等于size的2的整数次幂
		nextpowto2 <<= 1;
	for (int i = size; i < nextpowto2;i++) {
		x[i].re = 0;
		x[i].im = 0;
	}
	size = nextpowto2;
}
void butterflyComputation()//蝶形运算
{
	int i = 0, j = 0, k = 0, m = 0;
	complex q, y, z;
	reverse();
	for (i = 0; i < log(size) / log(2); i++)//蝶形运算的级数
	{
		m = 1 << i;//移位，每次都是2的指数的形式增加
		for (j = 0; j < size; j += 2 * m) //一组蝶形运算，每一组的蝶形因子乘数不同
		{
			for (k = 0; k < m; k++)//蝶形结点的距离  一个蝶形运算 每个组内的蝶形运算
			{
				multiply(x[k + j + m], W[size * k / 2 / m], &q);
				add(x[j + k], q, &y);
				subtract(x[j + k], q, &z);
				x[j + k] = y;
				x[j + k + m] = z;
			}
		}
	}
}

void readFromFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
		return; 
    }

    size = 0;
    while (fscanf(file, "%lf %lf", &x[size].im, &x[size].re)!=EOF) 
	{
        size++;
        if(size>=N) 
		{
            fprintf(stderr, "输入数据超过预设大小\n");
			return; 
        }
    }
    fclose(file);
}

void writeToFile(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "无法打开文件，尝试创建文件: %s\n", filename);
        file = fopen(filename, "w+"); // 如果无法打开文件，则尝试创建文件
        if (file == NULL) {
            fprintf(stderr, "无法创建文件: %s\n", filename);
            return;
        }
    }

    for (int i=0;i<size;i++)
    {
        fprintf(file,"%.6f\n",x[i].re); // 输出复数的实部到文件，并以空格隔开
        if (x[i].im >= MIN)
            fprintf(file,"%.6f\n",x[i].im); // 当复数的虚部大于MIN时，输出+ 虚部 j的形式到文件
        else if (fabs(x[i].im) < MIN)
            fprintf(file,"0.000000\n"); // 当虚部小于MIN时，输出0到文件
        else
            fprintf(file,"%.6f\n",x[i].im); // 上述两个条件除外的形式，输出 虚部 j的形式到文件
    }

    fclose(file);
}

int main() 
{
    readFromFile("fft_test_0.txt");
    addZero();
	printf("x(n)反序后的序列\n");
    twiddleFactor();
    butterflyComputation();
    printf("x(n)的基-2FFT结果X(k)\n");
    output();
    writeToFile("output.txt");
    return 0;
}
