#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#define nx 400
#define ny 400
#define nz 400
#define dx 2
#define dy 2
#define dz 2
#define lowBorderX -1.0
#define lowBorderY -1.0
#define lowBorderZ -1.0
#define a 100000
double phi(double x, double y, double z) {
	return x * x + y * y + z * z;
}
double ro(double x, double y, double z) {
	return 6 - a * phi(x, y, z);
}
int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int count = 0;
	MPI_Request messages[4];
	double hx = (double)dx / (nx - 1), hy = (double)dy / (ny - 1), hz = (double)dz / (nz - 1),
		sqrHx = hx * hx, sqrHy = hy * hy, sqrHz = hz * hz,
		someConst = 1 / (2 / sqrHx + 2 / sqrHy + 2 / sqrHz + a),
		localMax = 0, difference, max = 3,
		startTime;
	double** previousLattice = (double**)malloc((nx / size) * sizeof(double*));
	double** lattice = (double**)malloc((nx / size) * sizeof(double*));
	double** roro = (double**)malloc((nx / size) * sizeof(double*));
	for (int i = 0; i < nx / size; i++) {
		previousLattice[i] = (double*)calloc(ny * nz, sizeof(double));
		lattice[i] = (double*)calloc(ny * nz, sizeof(double));
		roro[i] = (double*)calloc(ny * nz, sizeof(double));
	}
	double* borders[2];
	borders[0] = (double*)malloc(ny * nz * sizeof(double));
	borders[1] = (double*)malloc(ny * nz * sizeof(double));
	for (int i = 0; i < nx / size; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				if ((i == 0 && rank == 0) || (i == nx / size - 1 && rank == size - 1) || j == 0 || j == ny - 1 || k == 0
					|| k == nz - 1) {
					lattice[i][j * nz + k] = phi((i + rank * (nx / size)) * hx + lowBorderX, j * hy + lowBorderY, k *
						hz + lowBorderZ);

				}
				roro[i][j * nz + k] = ro((i + rank * (nx / size)) * hx + lowBorderX, j * hy + lowBorderY, k * hz +
					lowBorderZ);
			}
		}
	}
	startTime = MPI_Wtime();
	while (max >= 0.00000001) {
		count++;
		for (int i = 0; i < nx / size; i++) {
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					previousLattice[i][j * nz + k] = lattice[i][j * nz + k];
				}
			}
		}
		if (rank > 0) {
			MPI_Isend(previousLattice[0], ny * nz, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
				&messages[0]);
			MPI_Irecv(borders[0], ny * nz, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD,
				&messages[2]);
		}
		if (rank < size - 1) {
			MPI_Isend(previousLattice[nx / size - 1], ny * nz, MPI_DOUBLE, rank + 1, 1,
				MPI_COMM_WORLD, &messages[1]);
			MPI_Irecv(borders[1], ny * nz, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
				&messages[3]);
		}
		for (int i = 1; i < nx / size - 1; i++) {
			for (int j = 1; j < ny - 1; j++) {
				for (int k = 1; k < nz - 1; k++) {
					lattice[i][j * nz + k] = someConst * ((previousLattice[i - 1][j * nz + k] + previousLattice[i + 1][j
						* nz + k]) / sqrHx
						+ (previousLattice[i][(j - 1) * nz + k] + previousLattice[i][(j + 1) * nz + k]) / sqrHy
						+ (previousLattice[i][j * nz + k - 1] + previousLattice[i][j * nz + k + 1]) / sqrHz
						- roro[i][j * nz + k]);
				}
			}
		}
		if (rank > 0) {
			MPI_Wait(&messages[0], MPI_STATUS_IGNORE);
			MPI_Wait(&messages[2], MPI_STATUS_IGNORE);
		}
		if (rank < size - 1) {
			MPI_Wait(&messages[1], MPI_STATUS_IGNORE);
			MPI_Wait(&messages[3], MPI_STATUS_IGNORE);
		}
		if (nx == size) {
			if (rank > 0 && rank < size - 1) {
				for (int j = 1; j < ny - 1; j++) {
					for (int k = 1; k < nz - 1; k++) {
						lattice[0][j * nz + k] = someConst * ((borders[0][j * nz + k] + borders[1][j * nz + k]) / sqrHx
							+ (previousLattice[0][(j - 1) * nz + k] + previousLattice[0][(j + 1) * nz + k]) / sqrHy
							+ (previousLattice[0][j * nz + k - 1] + previousLattice[0][j * nz + k + 1]) / sqrHz
							- roro[0][j * nz + k]);
					}

				}
			}
		}
		else {
			if (rank > 0) {
				for (int j = 1; j < ny - 1; j++) {
					for (int k = 1; k < nz - 1; k++) {
						lattice[0][j * nz + k] = someConst * ((borders[0][j * nz + k] + previousLattice[1][j * nz + k])
							/ sqrHx
							+ (previousLattice[0][(j - 1) * nz + k] + previousLattice[0][(j + 1) * nz + k]) / sqrHy
							+ (previousLattice[0][j * nz + k - 1] + previousLattice[0][j * nz + k + 1]) / sqrHz
							- roro[0][j * nz + k]);
					}
				}
			}
			if (rank < size - 1) {
				for (int j = 1; j < ny - 1; j++) {
					for (int k = 1; k < nz - 1; k++) {
						lattice[nx / size - 1][j * nz + k] = someConst * ((previousLattice[nx / size - 2][j * nz + k] +
							borders[1][j * nz + k]) / sqrHx
							+ (previousLattice[nx / size - 1][(j - 1) * nz + k] + previousLattice[nx / size - 1][(j + 1) *
								nz + k]) / sqrHy
							+ (previousLattice[nx / size - 1][j * nz + k - 1] + previousLattice[nx / size - 1][j * nz + k
								+ 1]) / sqrHz
							- roro[nx / size - 1][j * nz + k]);
					}
				}
			}
		}
		localMax = 0;
		for (int i = 0; i < nx / size; i++) {
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					difference = fabs(lattice[i][j * nz + k] - previousLattice[i][j * nz + k]);
					if (difference > localMax) {
						localMax = difference;
					}
				}
			}
		}
		MPI_Allreduce(&localMax, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	if (rank == 0) {
		printf("time: %g\n", MPI_Wtime() - startTime);
	}
	localMax = 0;
	for (int i = 0; i < nx / size; i++) {
		for (int j = 1; j < ny - 1; j++) {
			for (int k = 1; k < nz - 1; k++) {
				difference = fabs(lattice[i][j * nz + k]
					- phi((i + rank * (nx / size)) * hx + lowBorderX, j * hy + lowBorderY, k * hz + lowBorderZ));
				if (difference > localMax) {
					localMax = difference;
				}
			}
		}
	}
	MPI_Allreduce(&localMax, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("delta: %g\n", max);
		printf("iterations: %d\n\n", count);
	}
	for (int i = 0; i < nx / size; i++) {
		printf("x = %d for proc %d\n", i, rank);
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				printf("%g ", lattice[i][j * nz + k]);
			}
			printf("\n");
		}
		printf("\n");
	}
	MPI_Finalize();
	return 0;
}
