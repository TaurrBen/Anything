//fft.c ��-2��ʱ���ȡFFT
#include <stdio.h>
#define _USE_MATH_DEFINES//ʹ��math.h����Ħг���
#include <math.h>
#include <malloc.h>
#define N 65536//��󳤶�
#define MIN 0.00000000001//С��MIN��ֵ����Ϊ0
typedef struct {       //����һ���ṹ���ʾ����������
	double re;//ʵ��
	double im;//�鲿
}complex;
complex x[N], *W;   //x���������У�W����ת����
FILE *fp,*fp2;

int size = 0;   //�������ݳ���

void output()//�������
{
	for (int i = 0; i < size; i++)
	{
		printf("%.6f", x[i].re);//���������ʵ��
		if (x[i].im >= MIN)
			printf("+%.6fj\n", x[i].im);//���������鲿����MINʱ�����+ �鲿 j����ʽ
		else if (fabs(x[i].im) < MIN)
			printf("\n");//���鲿С��MINʱ�������鲿�������
		else
			printf("%.6fj\n", x[i].im);//�������������������ʽ����� �鲿 j����ʽ
	}
}

void reverse()//�������е�����
{
	complex temp;
	int i = 0, j = 0, k = 0;
	double t;
	for (i = 0; i < size; i++)
	{
		k = i;
		j = 0;
		t = (log(size) / log(2));//������еļ���
		while ((t--) > 0)//���ð�λ���Լ�ѭ��ʵ����λ�ߵ�
		{
			j = j << 1;
			j |= (k & 1);
			k = k >> 1;
		}
		if (j > i) //��x(n)����λ����
		{
			temp = x[i];
			x[i] = x[j];
			x[j] = temp;
		}
	}
	output();//�������������
}
void twiddleFactor()//��ת����
{
	W = (complex*)malloc(sizeof(complex) * size);//��ָ�����size�Ŀռ� size�����ݳ���
	for (int i = 0; i < size; i++)
	{
		W[i].re = cos(2 * M_PI / size * i);//ʹ��ŷ����ʽ��ʾ
		W[i].im = -1 * sin(2 * M_PI / size * i);
	}
}
void add(complex a, complex b, complex* c)//�����ӷ�
{
	c->re = a.re + b.re;
	c->im = a.im + b.im;
}
void subtract(complex a, complex b, complex* c)//�����ӷ�
{
	c->re = a.re - b.re;
	c->im = a.im - b.im;
}
void multiply(complex a, complex b, complex* c)//�����˷�
{
	c->re = a.re * b.re - a.im * b.im;
	c->im = a.re * b.im + a.im * b.re;
}
void addZero()//�������в���
{
	int nextpowto2 = 1;
	while (nextpowto2 < size)//�ҵ����ڵ���size��2����������
		nextpowto2 <<= 1;
	for (int i = size; i < nextpowto2;i++) {
		x[i].re = 0;
		x[i].im = 0;
	}
	size = nextpowto2;
}
void butterflyComputation()//��������
{
	int i = 0, j = 0, k = 0, m = 0;
	complex q, y, z;
	reverse();
	for (i = 0; i < log(size) / log(2); i++)//��������ļ���
	{
		m = 1 << i;//��λ��ÿ�ζ���2��ָ������ʽ����
		for (j = 0; j < size; j += 2 * m) //һ��������㣬ÿһ��ĵ������ӳ�����ͬ
		{
			for (k = 0; k < m; k++)//���ν��ľ���  һ���������� ÿ�����ڵĵ�������
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
        fprintf(stderr, "�޷����ļ�: %s\n", filename);
		return; 
    }

    size = 0;
    while (fscanf(file, "%lf %lf", &x[size].im, &x[size].re)!=EOF) 
	{
        size++;
        if(size>=N) 
		{
            fprintf(stderr, "�������ݳ���Ԥ���С\n");
			return; 
        }
    }
    fclose(file);
}

void writeToFile(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "�޷����ļ������Դ����ļ�: %s\n", filename);
        file = fopen(filename, "w+"); // ����޷����ļ������Դ����ļ�
        if (file == NULL) {
            fprintf(stderr, "�޷������ļ�: %s\n", filename);
            return;
        }
    }

    for (int i=0;i<size;i++)
    {
        fprintf(file,"%.6f\n",x[i].re); // ���������ʵ�����ļ������Կո����
        if (x[i].im >= MIN)
            fprintf(file,"%.6f\n",x[i].im); // ���������鲿����MINʱ�����+ �鲿 j����ʽ���ļ�
        else if (fabs(x[i].im) < MIN)
            fprintf(file,"0.000000\n"); // ���鲿С��MINʱ�����0���ļ�
        else
            fprintf(file,"%.6f\n",x[i].im); // �������������������ʽ����� �鲿 j����ʽ���ļ�
    }

    fclose(file);
}

int main() 
{
    readFromFile("fft_test_0.txt");
    addZero();
	printf("x(n)����������\n");
    twiddleFactor();
    butterflyComputation();
    printf("x(n)�Ļ�-2FFT���X(k)\n");
    output();
    writeToFile("output.txt");
    return 0;
}
