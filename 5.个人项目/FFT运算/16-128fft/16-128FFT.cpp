#include <stdio.h>
#define _USE_MATH_DEFINES//ʹ��math.h����Ħг���
#include <math.h>
#include <malloc.h>
#define N 65536//��󳤶�
#define MIN 0.00000000001//С��MIN��ֵ����Ϊ0
#define FFT_NUM 128
typedef struct {
	double re; // ʵ��
	double im; // �鲿
} complex;

complex x[N*FFT_NUM],*W; // ��������,W����ת����
double abs_values[N*FFT_NUM];

void output(complex* x,int size)//�������
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

void reverse(complex* x,int size)//�������е�����
{
	complex temp;
	int i = 0, j = 0, k = 0;
	double t;
	for (i = 0; i < size; i++)
	{
		k = i;
		j = 0;
		t = log(size) / log(2);//������еļ���  log(128) / log(2) = 7
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
//	output(x,size);//�������������
}

twiddleFactor()//��ת����
{
	W = (complex*)malloc(sizeof(complex) * FFT_NUM);//��ָ�����size�Ŀռ� size�����ݳ���
	for (int i = 0; i < FFT_NUM; i++)
	{
		W[i].re = cos(2 * M_PI / FFT_NUM * i);//ʹ��ŷ����ʽ��ʾ
		W[i].im = -1 * sin(2 * M_PI / FFT_NUM * i);
//		printf("W[%d]:%lf %lf\n",i,W[i].re,W[i].im);//��ӡ��ת���� 
	}
}

void Assign(complex* A, double re, double im) 
{
	A->re = re;
	A->im = im;
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

double abscomplex(complex a)//ȡģ 
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

void addZero(complex* x,int* size)//�������в���
{
	//��������ĩβ���� 
	int nextpowto2 = *size;
	while (nextpowto2 % 16 != 0)//�ҵ����ڵ���size��16�ı��� 
		nextpowto2 ++;
	for (int i = *size; i < nextpowto2;i++) {
		x[i].re = 0;
		x[i].im = 0;
	}
	*size = nextpowto2;
	
	//ÿ16λ������128λ
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
        fprintf(stderr, "�޷����ļ�: %s\n", filename);
		return 0; 
    }
    int size = 0;
    while (fscanf(file, "%lf %lf", &x[size].im, &x[size].re)!=EOF) 
	{
        size++;
        if(size>=N) 
		{
            fprintf(stderr, "�������ݳ���Ԥ���С\n");
			return 0; 
        }
    }
    fclose(file);
    return size;
}

void writeToFile(const char* filename,complex* x,int size) {
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


void butterflyComputation(complex* x,int size)//��������
{
	int i = 0, j = 0, k = 0, m = 0;
	complex q, y, z;
	reverse(x,size);
	for (i = 0; i < log(size) / log(2); i++)//��������ļ���  log(128) / log(2) = 7
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
	//	output(x,size);
}


int main() {
	
	int size = 0;
	int maxCircle;//4092��������ֵ��һ�� 
	int maxIndex; //128��������ֵ��λ�� 
	complex maxComplex;//����ֵ�ĸ��� 
	double maxComplex_abs;//����ֵ 
	double phase;
	double result; 
	
	size = readFromFile("fft_test_0.txt",x);
	addZero(x,&size);
//	printf("x(n)����������\n");
	twiddleFactor();
	for(int i = 0;i < size/FFT_NUM;i++){
		butterflyComputation(&x[i * FFT_NUM],FFT_NUM);
		//����ģֵ
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
			maxIndex = Index + 1;//IndexΪ�������� 
			maxComplex = Complex;
			maxCircle = i + 1;//iΪ0��ʼ������ 
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
	//���ֵΪ��4314.171837, �����ڵ� 343��ѭ���еĵ�9��������ȫ�ֵ�43785���������� 
	printf("Maximum modulus: %f,appearing in the %d round %d(%d),\n", abs_values[maxIndex - 1],maxCircle,maxIndex - (maxCircle - 1) * FFT_NUM,maxIndex);
	printf("Complex with maximum modulus: (%f, %f)\n", maxComplex.im, maxComplex.re);
	printf("Phase: %f\n", phase);
	printf("Result: %f\n", result);
    printf("x(n)�Ļ�-2FFT���X(k)\n");
    writeToFile("output.txt",x,size);
    return 0;
}
