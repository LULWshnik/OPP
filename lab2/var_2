#include "omp.h"
#include <iostream>

#define e 0.000001
#define MATRIX_SIZE 5500
#define t 0.001
using namespace std;

void right(double* A, double* b, double* xn, double* rightResult) {
#pragma omp for
	for (int j = 0; j < MATRIX_SIZE; ++j) {
		rightResult[j] = 0;
		for (int i = 0; i < MATRIX_SIZE; ++i) {
			rightResult[j] += A[i + j * MATRIX_SIZE] * xn[i];
		}
		rightResult[j] -= b[j];
	}
	return;
}

double sqrTop = 0;

bool flag(double* rightResult, double* b, double sqrB) {
	sqrTop = 0;
#pragma omp for reduction(+:sqrTop) 
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		sqrTop += rightResult[i] * rightResult[i];
	}

	return sqrt(sqrTop / sqrB) > e;
}

int main(int argc, char** argv) {
	double* Xn = new double[MATRIX_SIZE];
	double* A = new double[MATRIX_SIZE * MATRIX_SIZE];
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			A[j + i * MATRIX_SIZE] = 1.0;
			if (i == j) {
				++A[j + i * MATRIX_SIZE];
			}
		}
	}

	double* B = new double[MATRIX_SIZE];
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		B[i] = MATRIX_SIZE + 1;
	}

	double* rightResult = (double*)calloc(MATRIX_SIZE, sizeof(double));
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		rightResult[i] = (-1) * B[i];
	}

	double sqrB = 0;
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		sqrB += B[i] * B[i];
	}

	double startTime = omp_get_wtime();
#pragma omp parallel num_threads(1)
	{
		int exitCond = 0;
		while ((flag(rightResult, B, sqrB)) && (exitCond < 10000))
		{
#pragma omp for
			for (int i = 0; i < MATRIX_SIZE; ++i) {
				Xn[i] = Xn[i] - t * rightResult[i];
			}
			right(A, B, Xn, rightResult);
			++exitCond;
		}
	}
	double endTime = omp_get_wtime();


	double temp = fabs(Xn[0]);
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		if (temp < fabs(Xn[i])) {
			temp = fabs(Xn[i]);
		}
	}

	delete[] A;
	delete[] B;
	delete[] Xn;
	delete[] rightResult;
	cout << "Execution time: " << endTime - startTime << "\n";
	if (temp > 1 + e) {
		cout << "Problem: " << temp << "\n";
		return 1;
	}
	return 0;
}
