#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_SIZE 16

__global__ void matrix_multiply(float* matrixA, float* matrixB, float* resultMatrix, int M, int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    float sum = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < N) {
            tileA[ty][tx] = matrixA[row * N + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0;
        }
        if (col < N && t * TILE_SIZE + ty < N) {
            tileB[ty][tx] = matrixB[(t * TILE_SIZE + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        resultMatrix[row * N + col] = sum;
    }
}

int main(){
    int M = 200;
    int N = 100;
    float* host_matrixA, * host_matrixB, * host_resultMatrix;
    float* device_matrixA, * device_matrixB, * device_resultMatrix;
    int size_matrixA = M * N * sizeof(float);
    int size_matrixB = N * N * sizeof(float);
    int size_resultMatrix = M * N * sizeof(float);

    host_matrixA = (float*)malloc(size_matrixA);
    host_matrixB = (float*)malloc(size_matrixB);
    host_resultMatrix = (float*)malloc(size_resultMatrix);

    srand(time(NULL));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            host_matrixA[i * N + j] =rand() % 10 + 1;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            host_matrixB[i * N + j] =rand() % 10 + 1;
        }
    }

    cudaMalloc(&device_matrixA, size_matrixA);
    cudaMalloc(&device_matrixB, size_matrixB);
    cudaMalloc(&device_resultMatrix, size_resultMatrix);

    cudaMemcpy(device_matrixA, host_matrixA, size_matrixA, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrixB, host_matrixB, size_matrixB, cudaMemcpyHostToDevice);

    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrix_multiply<<<dimGrid, dimBlock>>>(device_matrixA, device_matrixB, device_resultMatrix, M, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_resultMatrix, device_resultMatrix, size_resultMatrix, cudaMemcpyDeviceToHost);

    printf("Execution time: %.2f ms\n", milliseconds);

    free(host_matrixA);
    free(host_matrixB);
    free(host_resultMatrix);
    cudaFree(device_matrixA);
    cudaFree(device_matrixB);
    cudaFree(device_resultMatrix);

    return 0;
}
