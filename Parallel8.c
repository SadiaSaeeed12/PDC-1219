#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 500  // Matrix size (N x N)
#define NUM_THREADS 8  // Number of threads

// Parallel Matrix Multiplication with Static Scheduling
void multiply_matrices_static(int A[N][N], int B[N][N], int C[N][N]) {
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Parallel Matrix Multiplication with Dynamic Scheduling
void multiply_matrices_dynamic(int A[N][N], int B[N][N], int C[N][N]) {
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2) schedule(dynamic, 10)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    int A[N][N], B[N][N], C[N][N];
    srand(time(0));

    // Initialize matrices A and B with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0;
        }
    }

    double start, end;

    // Measure execution time for Static Scheduling
    start = omp_get_wtime();
    multiply_matrices_static(A, B, C);
    end = omp_get_wtime();
    printf("Static Scheduling Time: %f seconds\n", end - start);

    // Measure execution time for Dynamic Scheduling
    start = omp_get_wtime();
    multiply_matrices_dynamic(A, B, C);
    end = omp_get_wtime();
    printf("Dynamic Scheduling Time: %f seconds\n", end - start);

    return 0;
}
