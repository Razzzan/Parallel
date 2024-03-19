#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define NUM_THREADS 4 

pthread_t threads[NUM_THREADS];
int buckets[NUM_THREADS][10]; 

void bucket_sort(int* array, int size) {
    clock_t start_time, end_time;
    double total_time;

    start_time = clock(); 

    int bucket_sizes[NUM_THREADS] = {0};

    for (int i = 0; i < size; i++) {
        int bucket_index = array[i] / ((1000 + NUM_THREADS - 1) / NUM_THREADS);
        if (bucket_index >= NUM_THREADS)  
            bucket_index = NUM_THREADS - 1;
        int index = __sync_fetch_and_add(&bucket_sizes[bucket_index], 1); 
        buckets[bucket_index][index] = array[i]; 
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        for (int j = 1; j < bucket_sizes[i]; j++) {
            int key = buckets[i][j];
            int k = j - 1;
            while (k >= 0 && buckets[i][k] > key) {
                buckets[i][k + 1] = buckets[i][k];
                k--;
            }
            buckets[i][k + 1] = key;
        }
    }

    int index = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        for (int j = 0; j < bucket_sizes[i]; j++) {
            array[index++] = buckets[i][j];
        }
    }

    end_time = clock(); 
    total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Total time taken: %f seconds\n", total_time);
}

int main() {
    int size = 1000;
    int *arr = malloc(size * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 1000; 
    }

    bucket_sort(arr, size);

    free(arr); 

    return 0;
}
