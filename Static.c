#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    int count = 0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = HEIGHT / size;
    int start_row = rank * chunk_size;
    int end_row = start_row + chunk_size;

    int image_chunk[chunk_size][WIDTH];
    struct complex c;

    for (int k = 0; k < 10; k++) {
        clock_t start_time = clock(); // Start measuring time

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image_chunk[i - start_row][j] = cal_pixel(c);
            }
        }

        clock_t end_time = clock(); // End measuring time

        double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        double avg_time;
        MPI_Reduce(&elapsed_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            avg_time /= size;
            printf("Execution time of trial [%d]: %f seconds\n", k, avg_time);
        }
    }

    int *all_image = NULL;
    if (rank == 0) {
        all_image = (int *)malloc(sizeof(int) * WIDTH * HEIGHT);
    }
    MPI_Gather(image_chunk, chunk_size * WIDTH, MPI_INT, all_image, chunk_size * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int image[HEIGHT][WIDTH];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < chunk_size; j++) {
                for (int k = 0; k < WIDTH; k++) {
                    image[i * chunk_size + j][k] = all_image[i * chunk_size * WIDTH + j * WIDTH + k];
                }
            }
        }
        save_pgm("mandelbrotStatic.pgm", image);
        free(all_image);
    }

    MPI_Finalize();
    return 0;
}
