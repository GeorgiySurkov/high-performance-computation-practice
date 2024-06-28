#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Произвольная функция f(x, y)
float f(int x, int y) {
    return sinf(x) * cosf(y);  // Пример функции, можно подставить любую другую
}

// Функция для вычисления частной производной по оси X
void computePartialDerivativeX(float *local_A, float *local_B, int local_width, int height, int rank, int size, int global_width, float dx) {
    int start_col = rank * local_width;
    int end_col = start_col + local_width;

    for (int i = 0; i < height; ++i) {
        for (int j = start_col; j < end_col; ++j) {
            if (j == 0 || j == global_width - 1) {
                local_B[i * local_width + (j - start_col)] = 0;  // Граничные условия, можно задавать по-другому
            } else {
                local_B[i * local_width + (j - start_col)] = (local_A[i * local_width + (j + 1 - start_col)] - local_A[i * local_width + (j - 1 - start_col)]) / (2 * dx);
            }
        }
    }
}

// Функция для инициализации массива данных
void initializeArray(float *local_A, int local_width, int height, int rank, int size, int global_width) {
    int start_col = rank * local_width;
    int end_col = start_col + local_width;

    for (int i = 0; i < height; ++i) {
        for (int j = start_col; j < end_col; ++j) {
            local_A[i * local_width + (j - start_col)] = f(i, j);
        }
    }
}

// Основная функция
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sizes[][2] = {{10, 10}, {100, 100}, {1000, 1000}, {10000, 10000}};
    float dx = 1.0f;

    for (int i = 0; i < 4; ++i) {
        int global_width = sizes[i][0];
        int height = sizes[i][1];
        int local_width = global_width / size;

        // Выделение памяти под локальные части массивов
        float *local_A = (float *)malloc(local_width * height * sizeof(float));
        float *local_B = (float *)malloc(local_width * height * sizeof(float));

        // Измерение времени инициализации данных
        double init_start_time = MPI_Wtime();
        initializeArray(local_A, local_width, height, rank, size, global_width);
        double init_end_time = MPI_Wtime();
        double init_time = (init_end_time - init_start_time) * 1000;  // в миллисекундах

        // Измерение времени вычисления частной производной
        double compute_start_time = MPI_Wtime();
        computePartialDerivativeX(local_A, local_B, local_width, height, rank, size, global_width, dx);
        double compute_end_time = MPI_Wtime();
        double compute_time = (compute_end_time - compute_start_time) * 1000;  // в миллисекундах

        if (rank == 0) {
            printf("Grid size: %d x %d\n", global_width, height);
            printf("Initialization time: %.3f ms\n", init_time);
            printf("Computation time: %.3f ms\n", compute_time);
            printf("------------------------------------\n");
        }

        free(local_A);
        free(local_B);
    }

    MPI_Finalize();

    return 0;
}
