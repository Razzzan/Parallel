#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <openacc.h>

// Kernel function for matrix multiplication
void matrix_multiply(float *matrixA, float *matrixB, float *resultMatrix, int M, int N) {
    #pragma acc parallel loop copyin(matrixA[0:M*N]) copyin(matrixB[0:K*N]) copyout(resultMatrix[0:M*K]) independent
    {
    # pragma acc region if(accelerate)
    {
    # pragma acc loop independent
    for (int i = 0; i < M; i ++)
    {
      # pragma acc loop independent
      for (int j = 0; j < N ; j ++ )
      {
          float sum = 0;
          for (int k = 0; k < N ; k ++ ) {
            sum += matrixA[i * N + k] * matrixB[k * N + j];
          }
          resultMatrix[i * N + j] = sum ;
      }
    }
    }
    }
}

double getElapsedTime(struct timeval start, struct timeval stop) {
    return (double)(stop.tv_sec - start.tv_sec) * 1000.0 +
           (double)(stop.tv_usec - start.tv_usec) / 1000.0;
}

int main() {
    int M = 500;
    int N =250;
    float *host_matrixA, *host_matrixB, *host_resultMatrix;

    // Allocate memory on host
    host_matrixA = (float*) malloc(M * N * sizeof(float));
    host_matrixB = (float*) malloc(N * N * sizeof(float));
    host_resultMatrix = (float*) malloc(M * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            host_matrixA[i * N + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            host_matrixB[i * N + j] = i - j;
        }
    }

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    // Perform matrix multiplication
    matrix_multiply(host_matrixA, host_matrixB, host_resultMatrix, M, N);

    gettimeofday(&stop, NULL);
    double elapsed_time = getElapsedTime(start, stop);

    // Print execution time in milliseconds
    printf("Execution time: %.2f ms\n", elapsed_time);

    // Free memory
    free(host_matrixA);
    free(host_matrixB);
    free(host_resultMatrix);

    return 0;
}
