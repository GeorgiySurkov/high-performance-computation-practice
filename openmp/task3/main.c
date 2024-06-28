#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Произвольная функция f(x, y)
float f(int x, int y) {
    return sinf(x) * cosf(y);  // Можно заменить на другую функцию.
}

void computePartialDerivativeX(float *A, float *B, int width, int height, float dx) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (j == 0 || j == width - 1) {
                B[i * width + j] = 0;  // Граничные условия, можно задавать по-другому
            } else {
                B[i * width + j] = (A[i * width + (j + 1)] - A[i * width + (j - 1)]) / (2 * dx);
            }
        }
    }
}

void initializeArray(float *A, int width, int height) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            A[i * width + j] = f(i, j);
        }
    }
}

void runPartialDerivative(int width, int height) {
    float *A = (float *)malloc(width * height * sizeof(float));
    float *B = (float *)malloc(width * height * sizeof(float));
    float dx = 1.0f;

    // Измерение времени инициализации и копирования данных
    double init_start = omp_get_wtime();
    initializeArray(A, width, height);
    double init_stop = omp_get_wtime();
    double init_time = (init_stop - init_start) * 1000;

    // Измерение времени выполнения OpenMP части
    double start = omp_get_wtime();
    computePartialDerivativeX(A, B, width, height, dx);
    double stop = omp_get_wtime();
    double kernel_time = (stop - start) * 1000;

    printf("Partial derivative on grid %dx%d computed.\n", width, height);
    printf("Initialization time: %.3f ms\n", init_time);
    printf("Kernel execution time: %.3f ms\n", kernel_time);

    free(A);
    free(B);
}

int main() {
    int sizes[][2] = {{10, 10}, {100, 100}, {1000, 1000}, {10000, 10000}};

    for (int i = 0; i < 4; ++i) {
        runPartialDerivative(sizes[i][0], sizes[i][1]);
    }

    return 0;
}
