#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Произвольная функция f(x, y)
__device__ float f(int x, int y) {
    return sinf(x) * cosf(y);  // Можно заменить на другую функцию.
}

// CUDA kernel для инициализации массива значениями функции f(x, y)
__global__ void initializeArray(float *A, int width, int height) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        A[i * width + j] = f(i, j);
    }
}

// CUDA kernel для вычисления частной производной по x
__global__ void computePartialDerivativeX(float *A, float *B, int width, int height, float dx) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        if (j == 0 || j == width - 1) {
            B[i * width + j] = 0;  // Граничные условия, можно задавать по-другому
        } else {
            B[i * width + j] = (A[i * width + (j + 1)] - A[i * width + (j - 1)]) / (2 * dx);
        }
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void runPartialDerivative(int width, int height) {
    int size = width * height;
    float *d_A, *d_B;
    float dx = 1.0f;

    // Измерение времени инициализации и копирования данных
    cudaEvent_t init_start, init_stop;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_stop);
    cudaEventRecord(init_start);

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeArray<<<blocksPerGrid, threadsPerBlock>>>(d_A, width, height);

    cudaEventRecord(init_stop);
    cudaEventSynchronize(init_stop);
    float init_time = 0;
    cudaEventElapsedTime(&init_time, init_start, init_stop);

    // Измерение времени выполнения CUDA kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    computePartialDerivativeX<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, width, height, dx);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Мы не копируем массив B обратно на хост, так как он не используется на хосте в этом примере
    std::cout << "Partial derivative on grid " << width << "x" << height << " computed." << std::endl;
    std::cout << "Initialization and memory copy time: " << init_time << " ms" << std::endl;
    std::cout << "Kernel execution time: " << kernel_time << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_stop);
}

int main() {
    int sizes[][2] = {{10, 10}, {100, 100}, {1000, 1000}, {10000, 10000}};

    for (auto &size : sizes) {
        runPartialDerivative(size[0], size[1]);
    }

    return 0;
}
