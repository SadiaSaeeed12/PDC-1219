#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 500  // Matrix size (N x N)

void multiply_matrices(int A[N][N], int B[N][N], int C[N][N]) {
    // Performing matrix multiplication (Sequential)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
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
        }
    }

    clock_t start = clock();  // Start timing

    // Perform sequential matrix multiplication
    multiply_matrices(A, B, C);

    clock_t end = clock();  // End timing
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Matrix multiplication completed in %f seconds.\n", elapsed_time);

    return 0;
}
