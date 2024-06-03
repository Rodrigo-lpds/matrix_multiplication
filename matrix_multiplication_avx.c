// execute to run: gcc -mavx -o matrix_multiplication_avx matrix_multiplication_avx.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

#define MATRIX_SIZE 512 // Valores : 32, 64, 128, 256, 512

void dgemm(size_t n, double *A, double *B, double *C)
{

    for (size_t i = 0; i < n; i += 4)

        for (size_t j = 0; j < n; j++)
        {
            __m256d c0 = _mm256_load_pd(C + i + j * n); /* c0 = C[i][j] */
            for (size_t k = 0; k < n; k++)
                c0 = _mm256_add_pd(c0, /* c0 += A[i][k]*B[k][j] */
                                   _mm256_mul_pd(_mm256_load_pd(A + i + k * n),
                                                 _mm256_broadcast_sd(B + k + j * n)));
            _mm256_store_pd(C + i + j * n, c0); /* C[i][j] = c0 */
        }
}

void buildMatrix(int n, double *P, double *Q, double *M)
{
    for (int i = 0; i < n * n; i++)
        P[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < n * n; i++)
        Q[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < n * n; i++)
        M[i] = 0.0;
}

int main()
{
    size_t n = MATRIX_SIZE;
    double *MatrizP = (double *)aligned_alloc(32, n * n * sizeof(double));
    double *MatrizQ = (double *)aligned_alloc(32, n * n * sizeof(double));
    double *MatrizM = (double *)aligned_alloc(32, n * n * sizeof(double));

    clock_t time;

    buildMatrix(MATRIX_SIZE, MatrizP, MatrizQ, MatrizM);
    time = clock();
    dgemm(MATRIX_SIZE, MatrizP, MatrizQ, MatrizM);
    time = clock() - time;

    printf("O tempo de execução do programa é de: %0.6Lf segundos\n",
           ((long double)time) / CLOCKS_PER_SEC);

    free(MatrizP);
    free(MatrizQ);
    free(MatrizM);

    return 0;
}
