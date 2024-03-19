#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define NUM_THREADS 4 
#define BUCKET_SIZE 10 
#define ARRAY_SIZE 1000
#define MAX_VALUE 1000 

int buckets[NUM_THREADS][BUCKET_SIZE];

int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

void sort_bucket(int* bucket, int size) {
    qsort(bucket, size, sizeof(int), compare); 
}

void bucket_sort(int* array, int size) {
    clock_t start_time, end_time;
    double total_time;
    int bucket_sizes[NUM_THREADS] = {0}; 
    
    #pragma omp parallel for shared(bucket_sizes)
    for (int i = 0; i < size; i++) {
        int tid = omp_get_thread_num();
        int bucket_index = array[i] / (MAX_VALUE / NUM_THREADS); 
        if (bucket_index >= NUM_THREADS)  
            bucket_index = NUM_THREADS - 1;
        int pos = __sync_fetch_and_add(&bucket_sizes[bucket_index], 1);
        buckets[bucket_index][pos] = array[i];
    }

    start_time = clock();
    #pragma omp parallel for
    for (int i = 0; i < NUM_THREADS; i++) {
        sort_bucket(buckets[i], bucket_sizes[i]);
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
    int arr[ARRAY_SIZE];
    srand(time(NULL));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand() % (MAX_VALUE + 1); 
    }

    bucket_sort(arr, ARRAY_SIZE);


    return 0;
}
