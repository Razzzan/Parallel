#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function for matrix multiplication
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
    int num_rows = 500;
    int num_columns = 250;
    float *host_matrixA, *host_matrixB, *host_resultMatrix;
    float *device_matrixA, *device_matrixB, *device_resultMatrix;

    int size_matrixA = num_rows * num_columns * sizeof(float);
    int size_matrixB = num_columns * num_columns * sizeof(float);
    int size_resultMatrix = num_rows * num_columns * sizeof(float);

    // Allocate memory on host
    host_matrixA = (float*) malloc(size_matrixA);
    host_matrixB = (float*) malloc(size_matrixB);
    host_resultMatrix = (float*) malloc(size_resultMatrix);

    // Initialize matrices
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_columns; j++) {
            host_matrixA[i * num_columns + j] = i + j;
        }
    }
    for (int i = 0; i < num_columns; i++) {
        for (int j = 0; j < num_columns; j++) {
            host_matrixB[i * num_columns + j] = i - j;
        }
    }

    // Allocate memory on device
    cudaMalloc(&device_matrixA, size_matrixA);
    cudaMalloc(&device_matrixB, size_matrixB);
    cudaMalloc(&device_resultMatrix, size_resultMatrix);

    // Copy data from host to device
    cudaMemcpy(device_matrixA, host_matrixA, size_matrixA, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrixB, host_matrixB, size_matrixB, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid_dim((num_rows + 15) / 16, (num_columns + 15) / 16, 1);
    dim3 block_dim(16, 16, 1);

    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel for matrix multiplication
    matrix_multiply<<<grid_dim, block_dim>>>(device_matrixA, device_matrixB, device_resultMatrix, num_rows, num_columns);

    // Record stop time and synchronize device
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(host_resultMatrix, device_resultMatrix, size_resultMatrix, cudaMemcpyDeviceToHost);

    // Print execution time in milliseconds
    printf("Execution time: %.2f ms\n", milliseconds);

    // Free memory
    free(host_matrixA);
    free(host_matrixB);
    free(host_resultMatrix);
    cudaFree(device_matrixA);
    cudaFree(device_matrixB);
    cudaFree(device_resultMatrix);

    return 0;
}
