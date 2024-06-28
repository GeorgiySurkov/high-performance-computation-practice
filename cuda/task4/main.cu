#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 16

// CUDA kernel для перемножения матриц
__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < N && col < N) {
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Функция для перемножения матриц и замера времени
void runMatrixMultiplication(int N) {
    size_t size = N * N * sizeof(float);

    // Выделение памяти на хосте
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Инициализация матриц случайными значениями
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Определение размеров блока и сетки
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Измерение времени выполнения CUDA kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // Копирование результата обратно на хост
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Matrix multiplication with size " << N << " x " << N << " took " << kernelTime << " ms" << std::endl;

    // Освобождение памяти
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int sizes[][2] = {{10, 10}, {100, 100}, {1000, 1000}, {10000, 10000}};

    for (int i = 0; i < 4; ++i) {
        runMatrixMultiplication(sizes[i][0]);
    }

    return 0;
}
