#include "mpi.h"
#include <iostream>

#define e 0.00000001
#define MATRIX_SIZE 10000
#define t 0.01
#define NORM 0
using namespace std;

void Counting(double* A, double* B, double* X, int* rows_at_process, double* rightResult, int* shift) {
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int thisRank = (rank + size) % size;
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < rows_at_process[rank]; ++j) {
			for (int k = shift[thisRank]; k < shift[thisRank] + rows_at_process[thisRank]; ++k) {
				rightResult[j] += A[k + j * MATRIX_SIZE] * X[k - shift[thisRank]];
			}
		}
		thisRank = (rank - i + size) % size;

		MPI_Sendrecv_replace(X, rows_at_process[0], MPI_DOUBLE, (rank + 1) % size, 0, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	}
	for (int j = 0; j < rows_at_process[rank]; ++j) {
		rightResult[j] -= B[j];
	}
	return;
}

bool flag(double* rightResult, double normB, int* rows_at_process, int rank) {
	double norm = 0;
	for (int i = 0; i < rows_at_process[rank]; ++i) {
		norm += rightResult[i] * rightResult[i];
	}
	double temp = 0;
	double tempB = 0;
	MPI_Allreduce(&norm, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&normB, &tempB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	temp = sqrt(temp / tempB);
	return temp > e;
}

int main(int argc, char* argv[])
{

	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double start;
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
		B[i] = MATRIX_SIZE + 1;
		Xn[i] = 0;
		normB += B[i] * B[i];
	}


	
	for (int i = 0; i < MATRIX_SIZE; ++i) { 

		X[i] = 0.0;
	}

	if (rank) {

		MPI_Send(&normB, 1, MPI_DOUBLE, 0, NORM, MPI_COMM_WORLD);
	}
	else{

		start = MPI_Wtime();

		for (int i = 1; i < size; ++i) {
			double tmp;
			MPI_Recv(&tmp, 1, MPI_DOUBLE, i, NORM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			normB += tmp;
		}

		normB = sqrt(normB);
	}
	
	double* rightResult = new double[rows_at_process[rank]];
	
	Counting(A, B, X, rows_at_process, rightResult, shift);

	while (flag(rightResult, normB, rows_at_process, rank)) {
		norm = 0.0;
		for (int i = 0; i < rows_at_process[rank]; ++i) {
			X[i] -= t * rightResult[i];
		}

		Counting(A, B, X, rows_at_process, rightResult, shift);

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
