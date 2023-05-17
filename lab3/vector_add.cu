#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define THREAD_BLOCK_SIZE 512
#define VECTOR_SIZE 100000

__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void vectorAdd_cpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    // generate vectors
    float *a = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    float *b = (float *)malloc(sizeof(float) * VECTOR_SIZE);
    float *c = (float *)malloc(sizeof(float) * VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // cpu version
    double start, end;
    double cpu_timer;
    start = clock();
    vectorAdd_cpu(a, b, c, VECTOR_SIZE);
    end = clock();
    cpu_timer = (end - start) / CLOCKS_PER_SEC;
    printf("cpu time: %f\n", cpu_timer);

    // allocate memory on device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(float) * VECTOR_SIZE);
    cudaMalloc((void **)&d_b, sizeof(float) * VECTOR_SIZE);
    cudaMalloc((void **)&d_c, sizeof(float) * VECTOR_SIZE);

    // copy data from host to device
    cudaMemcpy(d_a, a, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * VECTOR_SIZE, cudaMemcpyHostToDevice);

    // create and start gpu_timer by cuda_event
    float gpu_timer = 0;
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // launch kernel
    dim3 dimBlock(VECTOR_SIZE/THREAD_BLOCK_SIZE+1, 1);
    dim3 dimThread(THREAD_BLOCK_SIZE, 1);

    cudaEventRecord(start_event, 0);
    vectorAdd<<<dimBlock, dimThread>>>(d_a, d_b, d_c, VECTOR_SIZE);
    cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpu_timer, start_event, stop_event);
    printf("gpu time: %f\n", gpu_timer/1000);
    printf("Speedup: %f \n", 1000*cpu_timer/gpu_timer);

    // copy data from device to host
    cudaMemcpy(c, d_c, sizeof(float) * VECTOR_SIZE, cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

