#include "mpi.h"
#include <iostream>

#define MATRIX_SIZE 23
#define t 10e-7

int main(int argc, char* argv[])
{

	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double* A = NULL;
	double* X = new double[MATRIX_SIZE];
	double* B = new double[MATRIX_SIZE];
	double* Xn = NULL;
	double normB = 0.0;

	int* rows_at_process = new int[size];//массив кол-ва строк по процессам

	int shift = 0;
	int remain = MATRIX_SIZE % size;

	for (int i = 0; i < size; ++i) {
		rows_at_process[i] = i < remain ? MATRIX_SIZE / size + 1 : MATRIX_SIZE / size; //распределяем строки по процессам 
		if (i < rank) {
			shift += rows_at_process[i]; //номер первой строки матрицы, относительно всей матрицы
		}
	}

	A = new double[rows_at_process[rank] * MATRIX_SIZE]; //выделяем память под часть матрицы для процесса

	Xn = new double[rows_at_process[rank]]; //результат уможения матрицы на вектор для конкретного процесса

	for (int i = 0; i < rows_at_process[rank]; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			A[i * MATRIX_SIZE + j] = ((shift + i) == j) + 1;
		}
	}

	for (int i = 0; i < MATRIX_SIZE; ++i) {

		B[i] = MATRIX_SIZE + 1;
		X[i] = 0.0;

		if (!rank) {
			normB += B[i] * B[i];
		}
	}
  
  std::cout << rank << of << size;
  MPI_Finalize();
	delete[] A;
	delete[] B;
	delete[] X;
	delete[] rows_at_process;
	delete[] Xn;
	return 0;
}
