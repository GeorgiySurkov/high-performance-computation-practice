#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel для суммирования элементов массива
__global__ void sumArray(int *array, int *result, int n) {
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < n) {
        shared_data[tid] = array[index];
    } else {
        shared_data[tid] = 0;
    }

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void runSumArray(int n) {
    int *h_array, *d_array, *d_result;
    int h_result = 0;

    // Измерение времени инициализации и копирования данных
    cudaEvent_t init_start, init_stop;
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_stop);
    cudaEventRecord(init_start);

    h_array = (int *)malloc(n * sizeof(int));
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Инициализация массива случайными значениями
    for (int i = 0; i < n; ++i) {
        h_array[i] = rand() % 100;
    }

    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(init_stop);
    cudaEventSynchronize(init_stop);
    float init_time = 0;
    cudaEventElapsedTime(&init_time, init_start, init_stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Измерение времени выполнения CUDA kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sumArray<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_array, d_result, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start, stop);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum of array with " << n << " elements: " << h_result << std::endl;
    std::cout << "Initialization and memory copy time: " << init_time << " ms" << std::endl;
    std::cout << "Kernel execution time: " << kernel_time << " ms" << std::endl;

    free(h_array);
    cudaFree(d_array);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_stop);
}

int main() {
    int sizes[] = {10, 1000, 10000000};

    for (int size : sizes) {
        runSumArray(size);
    }

    return 0;
}
