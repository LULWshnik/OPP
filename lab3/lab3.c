#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n1 7200
#define n2 3333
#define n3 2400
#define p1 4
#define p2 4
void mulM(double* firstM, double* secondM, double* mOut) {
	for (int i = 0; i < n1 / p1; i++) {
		double* Out = mOut + i * n3 / p2;
		for (int k = 0; k < n2; k++) {
			const double* second = secondM + k * n3 / p2;
			double first = firstM[i * n2 + k];
			for (int j = 0; j < n3 / p2; j++)
				Out[j] += first * second[j];
		}
	}
}
int main(int argc, char** argv) {
	int size, rank;
	MPI_Init(&argc, &argv);
	double timer = MPI_Wtime();
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (size != p1 * p2) {
		printf("wrong number of procs");
		return -1;
	}
	MPI_Comm MPI_COMM_COLUMNS[p2], MPI_COMM_ROWS[p1];
	MPI_Group MPI_GROUP_WORLD, MPI_GROUP_COLUMNS[p2], MPI_GROUP_ROWS[p1];
	MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
	int ranksForColumns[p1];
	for (int i = 0; i < p2; i++) {
		for (int j = 0; j < p1; j++) {
			ranksForColumns[j] = j * p2 + i;
		}
		MPI_Group_incl(MPI_GROUP_WORLD, p1, ranksForColumns, &MPI_GROUP_COLUMNS[i]);
		MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_COLUMNS[i],
			&MPI_COMM_COLUMNS[i]);
	}
	int ranksForRows[p2];
	for (int i = 0; i < p1; i++) {
		for (int j = 0; j < p2; j++) {
			ranksForRows[j] = i * p2 + j;
		}
		MPI_Group_incl(MPI_GROUP_WORLD, p2, ranksForRows, &MPI_GROUP_ROWS[i]);
		MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_ROWS[i], &MPI_COMM_ROWS[i]);
	}
	double* mA = (double*)calloc((n1 * n2 / p1), sizeof(double));
	double* mB = (double*)calloc((n2 * n3 / p2), sizeof(double));
	double* mC = (double*)calloc((n1 / p1 * n3 / p2), sizeof(double));

	double* fullMA = NULL;
	double* fullMB = NULL;
	double* fullMC = NULL;
	if (rank == 0) {
		fullMA = (double*)malloc(n1 * n2 * sizeof(double));
		fullMB = (double*)malloc(n2 * n3 * sizeof(double));
		srand(777);
		for (int i = 0; i < n2; i++) {
			for (int j = 0; j < n1; j++) {
				fullMA[j * n2 + i] = (double)(rand() % 10000);
				if ((int)fullMA[j * n2 + i] % 3 == 0) {
					fullMA[j * n2 + i] *= -1;
				}
			}
			for (int j = 0; j < n3; j++) {
				fullMB[i * n3 + j] = (double)(rand() % 10000);
				if ((int)fullMB[i * n3 + j] % 3 == 0) {
					fullMB[i * n3 + j] *= -1;
				}
			}
		}
	}
	if (rank % p2 == 0) {
		MPI_Scatter(fullMA, n1 / p1 * n2, MPI_DOUBLE, mA, n1 / p1 * n2, MPI_DOUBLE, 0,
			MPI_COMM_COLUMNS[0]);
	}
	free(fullMA);
	MPI_Datatype MPI_COLUMN;
	MPI_Type_vector(n2, n3 / p2, n3, MPI_DOUBLE, &MPI_COLUMN);
	MPI_Type_create_resized(MPI_COLUMN, 0, n3 / p2 * sizeof(double), &MPI_COLUMN);
	MPI_Type_commit(&MPI_COLUMN);
	if (rank < p2) {
		MPI_Scatter(fullMB, 1, MPI_COLUMN, mB, n2 * n3 / p2, MPI_DOUBLE, 0,
			MPI_COMM_ROWS[0]);
	}
	MPI_Type_free(&MPI_COLUMN);
	free(fullMB);
	for (int i = 0; i < p2; i++) {
		if (rank % p2 == i) {
			MPI_Bcast(mB, n2 * n3 / p2, MPI_DOUBLE, 0, MPI_COMM_COLUMNS[i]);
		}
	}
	for (int i = 0; i < p1; i++) {
		if (rank < p2 * (i + 1) && rank >= p2 * i) {
			MPI_Bcast(mA, n2 * n1 / p1, MPI_DOUBLE, 0, MPI_COMM_ROWS[i]);
		}
	}
	mulM(mA, mB, mC);
	MPI_Datatype MPI_BLOCK;
	MPI_Type_vector(n1 / p1, n3 / p2, n3, MPI_DOUBLE, &MPI_BLOCK);
	MPI_Type_create_resized(MPI_BLOCK, 0, n3 / p2 * sizeof(double), &MPI_BLOCK);
	MPI_Type_commit(&MPI_BLOCK);
	int recvcounts[p1 * p2], displs[p1 * p2];
	for (int i = 0; i < p1 * p2; i++) {
		displs[i] = i % p2 + i / p2 * (n1 / p1 * p2);

		recvcounts[i] = 1;
	}
	if (rank == 0) {
		fullMC = (double*)malloc(n1 * n3 * sizeof(double));
	}
	MPI_Gatherv(mC, n1 * n3 / p1 / p2, MPI_DOUBLE, fullMC, recvcounts, displs, MPI_BLOCK, 0,
		MPI_COMM_WORLD);
	MPI_Type_free(&MPI_BLOCK);
	if (rank == 0) {
		printf("\ntime = %g\n", MPI_Wtime() - timer);
	}
	MPI_Finalize();
	free(mA);
	free(mB);
	free(mC);
	free(fullMC);
	return 0;
}
