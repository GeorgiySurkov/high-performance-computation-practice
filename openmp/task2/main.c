#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_THREADS 4

int sum_array(int *array, int size) {
    int sum = 0;
    int i;
#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    int sizes[] = {10, 1000, 10000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    srand(time(NULL));

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        int *array = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            array[i] = rand() % 100; // Заполняем массив случайными целыми числами от 0 до 99
        }

        double start_time = omp_get_wtime();
        int sum = sum_array(array, size);
        double end_time = omp_get_wtime();

        double kernel_time = (end_time - start_time) * 1000; // Время выполнения в миллисекундах

        printf("Size: %d, Sum: %d, Time: %f ms\n", size, sum, kernel_time);
        free(array);
    }

    return 0;
}
