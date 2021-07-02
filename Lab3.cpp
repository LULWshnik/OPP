#include<iostream>
#include<mpi.h>

#define NUM_DIMS 2
#define M 2300
#define N 2300
#define K 2300
#define A(i, j) A[N*i+j]
#define B(i, j) B[K*i+j]
#define C(i, j) C[K*i+j]
#define AA(i, j) AA[n[1]*i+j]
#define BB(i, j) BB[nn[1]*i+j]
#define CC(i, j) CC[nn[1]*i+j]
using namespace std;

void createTypes(int* n, int* nn, MPI_Datatype* typeb, MPI_Datatype* typec) {
	MPI_Datatype types;
	MPI_Type_vector(n[1], nn[1], n[2], MPI_DOUBLE, &types); //разбиение B

	MPI_Aint sizeofdouble; //тип нужный для адресов
	MPI_Type_extent(MPI_DOUBLE, &sizeofdouble); //узнали размер в байтах
	MPI_Type_create_resized(types, 0, sizeofdouble * nn[1], typeb); //выравнивание для более быстрого доступа к элементам
	MPI_Type_commit(typeb); //регистрируем новый производный тип

	MPI_Type_vector(nn[0], nn[1], n[2], MPI_DOUBLE, &types); //разбиение для С
	MPI_Type_create_resized(types, 0, sizeofdouble * nn[1], typec);
	MPI_Type_commit(typec);
}

void calculate(int* n, double* A, double* B, double* C, int* dims, MPI_Comm comm) {
	int* countc, * dispc, * countb, * dispb;
	MPI_Datatype typeb, typec;
	MPI_Comm pcomm;

	MPI_Comm_dup(comm, &pcomm); //скопировали в новый базовый коммуникатор 
	MPI_Bcast(n, 3, MPI_INT, 0, pcomm); //рассылка всем процессам матриц
	MPI_Bcast(dims, 2, MPI_INT, 0, pcomm);

	int periods[2] = { 0 };
	MPI_Comm comm_2D;
	MPI_Cart_create(pcomm, NUM_DIMS, dims, periods, 0, &comm_2D); //создаем коммуникатор с декартовой топологией

	int threadCoords[2];
	int threadRank;
	MPI_Comm_rank(comm_2D, &threadRank);
	MPI_Cart_coords(comm_2D, threadRank, NUM_DIMS, threadCoords); //узнали свою координату

	MPI_Comm comm_1D[2];
	int remains[2];

	for (int i = 0; i < 2; i++) { //делаем матрицу с главное диагональю елиницами
		for (int j = 0; j < 2; j++) {
			remains[j] = (i == j); // TF FT
		}
		MPI_Cart_sub(comm_2D, remains, &comm_1D[i]); //разделили коммуникатор на два одномерных коммуникатора с одномерными решетками
	}

	int nn[2];
	nn[0] = n[0] / dims[0]; //количество строк в полоске 
	nn[1] = n[2] / dims[1];

	double* AA, * BB, * CC;
	AA = new double[nn[0] * n[1]];
	BB = new double[n[1] * nn[1]];
	CC = new double[nn[0] * nn[1]];

	if (threadRank == 0) {
		createTypes(n, nn, &typeb, &typec); //создали типы с B и С

		dispb = new int[dims[1]]; //количество полосок
		countb = new int[dims[1]];
		dispc = new int[dims[0] * dims[1]];
		countc = new int[dims[0] * dims[1]];
		for (int j = 0; j < dims[1]; j++) {
			dispb[j] = j;
			countb[j] = 1;
		}

		for (int i = 0; i < dims[0]; i++) {
			for (int j = 0; j < dims[1]; j++) {
				dispc[i * dims[1] + j] = (i * dims[1] * nn[0] + j);
				countc[i * dims[1] + j] = 1;
			}
		}
	}

	if (threadCoords[1] == 0) { //коорд по оy равно 0
		MPI_Scatter(A, nn[0] * n[1], MPI_DOUBLE, AA, nn[0] * n[1], MPI_DOUBLE, 0, comm_1D[0]); //разбили А и разослали по потокам
	}

	if (threadCoords[0] == 0) {
		MPI_Scatterv(B, countb, dispb, typeb, BB, n[1] * nn[1], MPI_DOUBLE, 0, comm_1D[1]); //разбили В и разослали по потокам	
	}

	MPI_Bcast(AA, nn[0] * n[1], MPI_DOUBLE, 0, comm_1D[1]); //по другой оси, чтобы посчитать часть каждую

	MPI_Bcast(BB, n[1] * nn[1], MPI_DOUBLE, 0, comm_1D[0]);

	for (int i = 0; i < nn[0]; i++) {
		for (int j = 0; j < nn[1]; j++) {
			for (int k = 0; k < n[1]; k++) {
				CC(i, j) = CC(i, j) + AA(i, k) * BB(k, j); //5 шаг (каждый процесс высчитывает свою подматрицу)
			}
		}
	}

	MPI_Gatherv(CC, nn[0] * nn[1], MPI_DOUBLE, C, countc, dispc, typec, 0, comm_2D);

	delete[] AA;
	delete[] BB;
	delete[] CC;
	MPI_Comm_free(&pcomm);
	MPI_Comm_free(&comm_2D);
	for (int i = 0; i < 2; i++) {
		MPI_Comm_free(&comm_1D[i]);
	}


	if (threadRank == 0) {
		delete[] countc;
		delete[] dispc;
		delete[] countb;
		delete[] dispb;
		MPI_Type_free(&typeb);
		MPI_Type_free(&typec);
	}
}

int checkResult(double* C) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < K; ++j) {
			if (C(i, j) != N) {
				return 0;
			}
		}
	}
	return 1;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int threadCount;
	int threadRank;
	MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
	MPI_Comm_rank(MPI_COMM_WORLD, &threadRank);

	int dims[NUM_DIMS] = { 0 };  //количество узлов на каждой из осей решетки
	MPI_Dims_create(threadCount, NUM_DIMS, dims); //формирование решетки

	int n[3];
	double* A;
	double* B;
	double* C;
	if (threadRank == 0) {
		n[0] = M;
		n[1] = N;
		n[2] = K;

		A = new double[M * N];
		B = new double[N * K];
		C = new double[M * K];

		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				A(i, j) = 1;
			}
		}

		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < K; ++k) {
				B(j, k) = 1;
			}
		}
	}

	double startTime = MPI_Wtime();
	calculate(n, A, B, C, dims, MPI_COMM_WORLD);
	double finishTime = MPI_Wtime();

	if (threadRank == 0) {
		cout << "Time: " << finishTime - startTime << "\n"; 
		cout << "Result: ";
		if (checkResult(C))
			cout << "success";
		else
			cout << "failure";
		delete[] A;
		delete[] B;
		delete[] C;
	}

	MPI_Finalize();
	return 0;
}
