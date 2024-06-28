#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void matrixMultiplication(float* A, float* B, float* C, int N, int local_N) {
    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void runMatrixMultiplication(int N) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_N = N / size;
    size_t size_matrix = N * N * sizeof(float);
    size_t size_local_matrix = local_N * N * sizeof(float);

    float *A, *B, *C, *local_A, *local_C;
    A = B = C = local_A = local_C = NULL;

    if (rank == 0) {
        A = (float*)malloc(size_matrix);
        B = (float*)malloc(size_matrix);
        C = (float*)malloc(size_matrix);
        initializeMatrix(A, N);
        initializeMatrix(B, N);
    }

    local_A = (float*)malloc(size_local_matrix);
    local_C = (float*)malloc(size_local_matrix);

    MPI_Scatter(A, local_N * N, MPI_FLOAT, local_A, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    matrixMultiplication(local_A, B, local_C, N, local_N);
    double end_time = MPI_Wtime();

    MPI_Gather(local_C, local_N * N, MPI_FLOAT, C, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix multiplication with size %d x %d took %f ms\n", N, N, (end_time - start_time) * 1000);
        free(A);
        free(B);
        free(C);
    }

    free(local_A);
    free(local_C);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int sizes[][2] = {{10, 10}, {100, 100}, {1000, 1000}, {10000, 10000}};

    for (int i = 0; i < 4; ++i) {
        runMatrixMultiplication(sizes[i][0]);
    }

    MPI_Finalize();
    return 0;
}
