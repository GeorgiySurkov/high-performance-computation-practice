#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Функция для инициализации матриц случайными значениями
void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Функция для перемножения матриц с использованием OpenMP
void matrixMultiplication(float* A, float* B, float* C, int N) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Функция для выполнения перемножения матриц и замера времени
void runMatrixMultiplication(int N) {
    size_t size = N * N * sizeof(float);

    // Выделение памяти для матриц
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    // Инициализация матриц
    initializeMatrix(A, N);
    initializeMatrix(B, N);

    // Замер времени выполнения
    double start_time = omp_get_wtime();
    matrixMultiplication(A, B, C, N);
    double end_time = omp_get_wtime();

    printf("Matrix multiplication with size %d x %d took %f ms\n", N, N, (end_time - start_time) * 1000);

    // Освобождение памяти
    free(A);
    free(B);
    free(C);
}

int main() {
    int sizes[][2] = {{10, 10}, {100, 100}, {1000, 1000}, {10000, 10000}};

    for (int i = 0; i < 4; ++i) {
        runMatrixMultiplication(sizes[i][0]);
    }

    return 0;
}
