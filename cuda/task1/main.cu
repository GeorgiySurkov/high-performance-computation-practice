#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    printf("hello from cpu\n");
    cuda_hello<<<1,32>>>(); // 1 блок потоков, 32 потока в каждом
    cudaDeviceSynchronize();
    return 0;
}
