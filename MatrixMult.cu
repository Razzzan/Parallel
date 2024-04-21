%%cuda

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrix_multiply(float *matrixA, float *matrixB, float *resultMatrix, int rows, int columns){
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    int col_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_index < rows && col_index < columns) {
        float sum = 0.0;
        for (int k = 0; k < columns; k++) {
            sum += matrixA[row_index * columns + k] * matrixB[k * columns + col_index];
        }
        resultMatrix[row_index * columns + col_index] = sum;
    }
}

int main(){
    int num_rows = 100;
    int num_columns = 50;
    float *host_matrixA, *host_matrixB, *host_resultMatrix;
    float *device_matrixA, *device_matrixB, *device_resultMatrix;

    int size_matrixA = num_rows * num_columns * sizeof(float);
    int size_matrixB = num_columns * num_columns * sizeof(float);
    int size_resultMatrix = num_rows * num_columns * sizeof(float);

    // Allocate memory on host
    host_matrixA = (float*) malloc(size_matrixA);
    host_matrixB = (float*) malloc(size_matrixB);
    host_resultMatrix = (float*) malloc(size_resultMatrix);

    srand(time(NULL));
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_columns; j++) {
            host_matrixA[i * num_columns + j] = rand() % 10 + 1;
        }
    }
    for (int i = 0; i < num_columns; i++) {
        for (int j = 0; j < num_columns; j++) {
            host_matrixB[i * num_columns + j] = rand() % 10 + 1;
        }
    }

    cudaMalloc(&device_matrixA, size_matrixA);
    cudaMalloc(&device_matrixB, size_matrixB);
    cudaMalloc(&device_resultMatrix, size_resultMatrix);

    cudaMemcpy(device_matrixA, host_matrixA, size_matrixA, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrixB, host_matrixB, size_matrixB, cudaMemcpyHostToDevice);

    dim3 grid_dim((num_rows + 15) / 16, (num_columns + 15) / 16, 1);
    dim3 block_dim(16, 16, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrix_multiply<<<grid_dim, block_dim>>>(device_matrixA, device_matrixB, device_resultMatrix, num_rows, num_columns);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

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
