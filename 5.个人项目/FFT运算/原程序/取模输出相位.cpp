#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <malloc.h>
#define N 32 * 4092
#define MIN 0.00000000001

typedef struct {
	double re; // 实部
	double im; // 虚部
} complex;

complex x[N]; // 输入序列

int size = 0;

void readFromFile(const char* filename) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "无法打开文件: %s\n", filename);
		return;
	}
	size = 0;
	while (fscanf(file, "%lf %lf", &x[size].re, &x[size].im) != EOF) {
		size++;
		if (size >= N) {
			fprintf(stderr, "输入数据超过预设大小\n");
			return;
		}
	}
	fclose(file);
}

void Assign(complex* A, double re, double im) {
	A->re = re;
	A->im = im;
}

double abscomplex(complex a) {
	return sqrt(a.re * a.re + a.im * a.im);
}

int getMaxIndex(const double arr[], int size) {
	int maxIndex = 0;
	for (int i = 1; i < size; i++) {
		if (arr[i] > arr[maxIndex]) {
			maxIndex = i;
		}
	}
	return maxIndex;
}

int main() {
	double complex_abs;
	complex x1;
	readFromFile("output.txt");
	double abs_values[N];
	for (int i = 0; i < size; i++) {
		Assign(&x1, x[i].re, x[i].im);
		complex_abs = abscomplex(x1);
		abs_values[i] = complex_abs;
		printf("abs of the complex is: %f\n", complex_abs);
	}

	int maxIndex = getMaxIndex(abs_values, size);
	complex maxComplex = x[maxIndex];
	double phase;

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

	double result = (16.0 / 128.0) * (1000.0 * phase);
	printf("Maximum modulus: %f\n", abs_values[maxIndex]);
	printf("Complex with maximum modulus: (%f, %f)\n", maxComplex.im, maxComplex.re);
	printf("Phase: %f\n", phase);
	printf("Result: %f\n", result);
	return 0;
}
