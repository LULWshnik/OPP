#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <stddef.h>
#define numOfIterations 30
typedef struct List_s {
	int repeatNum;
	struct List_s* next;
} List;
void push(List* head, int repeatNum) {
	List* p = head;
	while (p->next != NULL) {
		p = p->next;
	}
	p->next = (List*)malloc(sizeof(List));
	p->next->repeatNum = repeatNum;
	p->next->next = NULL;
}
List* createList(int i) {
	int rank, size;
	int L = 1000000;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	List* head = (List*)malloc(sizeof(List));
	head->repeatNum = (numOfIterations / 2) * abs(rank - (i % size)) * L;
	head->next = NULL;
	for (int j = 1; j < numOfIterations; j++) {
		push(head, abs((numOfIterations / 2) - j % numOfIterations) * abs(rank - (i % size)) * L);
	}
	return head;
}
int pop(List* head) {
	int val;
	List* p = head;
	if (p->next == NULL) {
		val = p->repeatNum;
		free(p);
		return val;
	}
	while (p->next->next != NULL) {
		p = p->next;
	}

	val = p->next->repeatNum;
	free(p->next);
	p->next = NULL;
	return val;
}
void freeList(List* head) {
	List* p;
	while (head != NULL) {
		p = head;
		head = head->next;
		free(p);
	}
}
int taskToDoLeft;
List* taskToDo;
pthread_mutex_t mut;
void rebalance() {
	int temp = 0;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	for (int i = 0; i < size; i++) {
		MPI_Send(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	}
	for (int i = 0; i < size; i++) {
		MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (temp != 0) {
			taskToDoLeft++;
			push(taskToDo, temp);
		}
		MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (temp != 0) {
			taskToDoLeft++;
			push(taskToDo, temp);
		}
	}
}
void* supportThreadFunc(void* arg) {
	int rank, size;
	int endFlag;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status stat;
	while (1) {
		MPI_Recv(&endFlag, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
		if (endFlag) {
			break;
		}
		int valueFirst = 0, valueSec = 0, max = 0;

		pthread_mutex_lock(&mut);
		MPI_Allreduce(&taskToDoLeft, &max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		if (taskToDoLeft > 0 && taskToDoLeft == max) {
			taskToDoLeft--;
			valueFirst = pop(taskToDo);
			if (taskToDoLeft > 0) {
				taskToDoLeft--;
				valueSec = pop(taskToDo);
			}
		}
		pthread_mutex_unlock(&mut);
		MPI_Send(&valueFirst, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
		MPI_Send(&valueSec, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
	}
}
int main(int argc, char** argv) {
	int temp;
	int rank, size;
	int trueFlag = 1;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &temp);
	if (temp != MPI_THREAD_MULTIPLE) {
		perror("there is no thread safety\n");
		exit(-1);
	}
	pthread_mutex_init(&mut, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	pthread_t supportThread;
	pthread_create(&supportThread, NULL, supportThreadFunc, NULL);
	for (int i = 0; i < 3; i++) {
		taskToDo = createList(i);
		List* currentTask = taskToDo;
		List* lastTask;
		double localRes = 0;
		taskToDoLeft = numOfIterations;
		MPI_Barrier(MPI_COMM_WORLD);
		double timer = MPI_Wtime();
		while (taskToDoLeft > 0) {
			while (currentTask) {
				pthread_mutex_lock(&mut);
				for (int j = 0; j < currentTask->repeatNum; j++) {
					localRes += sin(j);
				}
				taskToDoLeft--;
				pthread_mutex_unlock(&mut);
				sleep(1);
				if (currentTask->next == NULL) {
					lastTask = currentTask;
				}
				currentTask = currentTask->next;
			}
			rebalance();
			currentTask = lastTask->next;

		}
		printf("for proc%d time is %g\n", rank, MPI_Wtime() - timer);
		printf("for proc%d localRes is %g\n", rank, localRes);
		MPI_Barrier(MPI_COMM_WORLD);
		freeList(taskToDo);
	}
	MPI_Send(&trueFlag, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
	pthread_join(supportThread, NULL);
	pthread_mutex_destroy(&mut);
	MPI_Finalize();
	return 0;
}
