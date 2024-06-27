#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sizes[] = {10, 1000, 10000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    srand(time(NULL) + rank);

    for (int s = 0; s < num_sizes; s++) {
        int array_size = sizes[s];
        int *array = NULL;
        int *sub_array = NULL;
        int base_sub_array_size = array_size / size;
        int extra_elements = array_size % size;

        if (rank == 0) {
            array = (int *)malloc(array_size * sizeof(int));
            for (int i = 0; i < array_size; i++) {
                array[i] = rand() % 100; // Заполняем массив случайными целыми числами от 0 до 99
            }
        }

        int *sendcounts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            sendcounts[i] = base_sub_array_size;
            if (i < extra_elements) {
                sendcounts[i]++;
            }
            displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
        }

        sub_array = (int *)malloc(sendcounts[rank] * sizeof(int));

        MPI_Scatterv(array, sendcounts, displs, MPI_INT, sub_array, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

        double start_time = MPI_Wtime();
        int local_sum = 0;
        for (int i = 0; i < sendcounts[rank]; i++) {
            local_sum += sub_array[i];
        }

        int global_sum = 0;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        if (rank == 0) {
            double kernel_time = (end_time - start_time) * 1000; // Время выполнения в миллисекундах
            printf("Size: %d, Sum: %d, Time: %f ms\n", array_size, global_sum, kernel_time);
            free(array);
        }

        free(sub_array);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
