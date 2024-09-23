#include <stdio.h>
#define _USE_MATH_DEFINES//使用math.h里面的π常量
#include <math.h>
#include <malloc.h>
#define N 65536//最大长度
#define MIN 0.00000000001//小于MIN的值被视为0
#define FFT_NUM 128
typedef struct {
	double re; // 实部
	double im; // 虚部
} complex;

complex x[N*FFT_NUM],*W; // 输入序列,W是旋转因子
double abs_values[N*FFT_NUM];

void output(complex* x,int size)//输出复数
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

void reverse(complex* x,int size)//输入序列的整序
{
	complex temp;
	int i = 0, j = 0, k = 0;
	double t;
	for (i = 0; i < size; i++)
	{
		k = i;
		j = 0;
		t = log(size) / log(2);//算出序列的级数  log(128) / log(2) = 7
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
//	output(x,size);//输出整序后的序列
}

twiddleFactor()//旋转因子
{
	W = (complex*)malloc(sizeof(complex) * FFT_NUM);//给指针分配size的空间 size是数据长度
	for (int i = 0; i < FFT_NUM; i++)
	{
		W[i].re = cos(2 * M_PI / FFT_NUM * i);//使用欧拉公式表示
		W[i].im = -1 * sin(2 * M_PI / FFT_NUM * i);
//		printf("W[%d]:%lf %lf\n",i,W[i].re,W[i].im);//打印旋转因子 
	}
}

void Assign(complex* A, double re, double im) 
{
	A->re = re;
	A->im = im;
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

double abscomplex(complex a)//取模 
{
	return sqrt(a.re * a.re + a.im * a.im);
}

int getMaxIndex(double arr[], int size,double *max) {
	int maxIndex = 0;
	for (int i = 1; i < size; i++) {
		if (arr[i] > arr[maxIndex]) {
			*max = arr[i]; 
			maxIndex = i;
		}
	}
	return maxIndex;
}

void addZero(complex* x,int* size)//输入序列补零
{
	//整个序列末尾补零 
	int nextpowto2 = *size;
	while (nextpowto2 % 16 != 0)//找到大于等于size的16的倍数 
		nextpowto2 ++;
	for (int i = *size; i < nextpowto2;i++) {
		x[i].re = 0;
		x[i].im = 0;
	}
	*size = nextpowto2;
	
	//每16位后补零至128位
	complex* y = (complex*)malloc(sizeof(complex) * (N*128));
	int j = 0;
	for(int i = 0;i < *size;i++,j++){ 
		y[j] = x[i];
		if((i + 1) % 16 == 0){
			do{
				y[j + 1].re = 0;
				y[j + 1].im = 0;
				j++;
			}while(j % 128 != 127); 
		}
	}
	*size = j;
	for(int i = 0;i < *size;i++)
	{
		x[i]= y[i];
	}
}


int readFromFile(const char* filename,complex* x) {
	FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
		return 0; 
    }
    int size = 0;
    while (fscanf(file, "%lf %lf", &x[size].im, &x[size].re)!=EOF) 
	{
        size++;
        if(size>=N) 
		{
            fprintf(stderr, "输入数据超过预设大小\n");
			return 0; 
        }
    }
    fclose(file);
    return size;
}

void writeToFile(const char* filename,complex* x,int size) {
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


void butterflyComputation(complex* x,int size)//蝶形运算
{
	int i = 0, j = 0, k = 0, m = 0;
	complex q, y, z;
	reverse(x,size);
	for (i = 0; i < log(size) / log(2); i++)//蝶形运算的级数  log(128) / log(2) = 7
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
	//	output(x,size);
}


int main() {
	
	int size = 0;
	int maxCircle;//4092次中最大幅值的一次 
	int maxIndex; //128次中最大幅值的位置 
	complex maxComplex;//最大幅值的复数 
	double maxComplex_abs;//最大幅值 
	double phase;
	double result; 
	
	size = readFromFile("fft_test_0.txt",x);
	addZero(x,&size);
//	printf("x(n)反序后的序列\n");
	twiddleFactor();
	for(int i = 0;i < size/FFT_NUM;i++){
		butterflyComputation(&x[i * FFT_NUM],FFT_NUM);
		//计算模值
		for (int j = 0; j < FFT_NUM; j++) {
			int temp = j + i * FFT_NUM;
			complex x1;
			double complex_abs;
			Assign(&x1, x[temp].re, x[temp].im);
			complex_abs = abscomplex(x1);
			abs_values[temp] = complex_abs;
		}
		
		double Complex_abs = 0;
		int Index = getMaxIndex(&abs_values[i * FFT_NUM], FFT_NUM,&Complex_abs) + i * FFT_NUM;
		complex Complex = x[Index];
		if(maxComplex_abs <= Complex_abs){
			maxComplex_abs = Complex_abs;
			maxIndex = Index + 1;//Index为数组索引 
			maxComplex = Complex;
			maxCircle = i + 1;//i为0起始的索引 
			if (maxComplex.re > MIN || maxComplex.re < -MIN) {
				phase = atan(maxComplex.im / maxComplex.re);
			}
			else if (maxComplex.im > MIN) {
				phase = M_PI / 2.0;
			}
			else if (maxComplex.im < -MIN) {
				phase = -M_PI / 2.0;
			}
			else {
				phase = 0.0; // undefined
			}
			result = (16.0 / 128.0) * (1000.0 * phase);
		} 
	}
	//最大值为：4314.171837, 出现在第 343次循环中的第9个复数（全局第43785个复数）。 
	printf("Maximum modulus: %f,appearing in the %d round %d(%d),\n", abs_values[maxIndex - 1],maxCircle,maxIndex - (maxCircle - 1) * FFT_NUM,maxIndex);
	printf("Complex with maximum modulus: (%f, %f)\n", maxComplex.im, maxComplex.re);
	printf("Phase: %f\n", phase);
	printf("Result: %f\n", result);
    printf("x(n)的基-2FFT结果X(k)\n");
    writeToFile("output.txt",x,size);
    return 0;
}
