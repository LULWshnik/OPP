#include "mpi.h"
#include <iostream>

#define e 0.00000001
#define MATRIX_SIZE 10000
#define t 0.01
using namespace std;

bool flag(double norm, double normB, int rank) {
	double temp = 0;
	MPI_Allreduce(&norm, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	temp = sqrt(temp / normB);
	return temp > e;
}

int main(int argc, char* argv[])
{

	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double start = MPI_Wtime();
	double* A = NULL;
	double* X = new double[MATRIX_SIZE];
	double* B = new double[MATRIX_SIZE];
	double norm;
	double normB = 0.0;

	int* rows_at_process = new int[size];//массив кол-ва строк по процессам

	for (int i = 0; i < size; ++i) {
		rows_at_process[i] = (MATRIX_SIZE / size) + ((i < MATRIX_SIZE % size) ? (1) : (0));//распределяем строки по процессам 
	}

	int* shift = new int[size];
	fill(shift, shift + size, 0);
	for (int i = 1; i < size; ++i) {
		shift[i] = shift[i - 1] + rows_at_process[i - 1];
	}

	A = new double[rows_at_process[rank] * MATRIX_SIZE]; //выделяем память под часть матрицы для процесса

	double* Xn = new double[rows_at_process[rank]]; //результат уможения матрицы на вектор для конкретного процесса

	for (int i = 0; i < rows_at_process[rank]; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			A[i * MATRIX_SIZE + j] = (shift[rank] + 1 == j) + 1;
		}
	}
	
	for (int i = 0; i < MATRIX_SIZE; ++i) { //заполняем B
	
		B[i] = MATRIX_SIZE + 1;
		X[i] = 0.0;

		if (!rank) {
			normB += B[i] * B[i];
		}
	}
	
	double* rightResult = new double[rows_at_process[rank]];
	for (int i = 0; i < rows_at_process[rank]; ++i) { 
		rightResult[i] = (-1) * B[i + shift[rank]];
		norm = rightResult[i] * rightResult[i];
	}
	int exitCond = 0;

	while (flag(norm, normB, rank)) {
		norm = 0.0;
		for (int i = 0; i < rows_at_process[rank]; ++i) {
			Xn[i] = X[i + shift[rank]] - t * rightResult[i];
		}

		MPI_Allgatherv(Xn, rows_at_process[rank], MPI_DOUBLE, X, rows_at_process, shift, MPI_DOUBLE, MPI_COMM_WORLD); // собираем новый x от всех процессов

		for (int j = 0; j < rows_at_process[rank]; ++j) {
			double result = 0;
			for (int i = 0; i < MATRIX_SIZE; ++i) {
				result += A[i + j * MATRIX_SIZE] * X[i];
			}
			rightResult[j] = result - B[j];
			norm = rightResult[j] * rightResult[j];
		}
	}

	free(shift);
	delete[] A;
	delete[] B;
	delete[] X;
	delete[] rightResult;
	delete[] rows_at_process;
	delete[] Xn;

	if (!rank) {
		double end = MPI_Wtime();
		cout << "Execution time: " << end - start << "\n";

	}
	
	MPI_Finalize();

	return 0;
}
