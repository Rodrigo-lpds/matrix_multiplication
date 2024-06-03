// execute to run: gcc -O3 -mavx512f -o matrix_multiplication_cache matrix_multiplication_cache.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

#define MATRIX_SIZE 512 // Valores : 32, 64, 128, 256, 512
#define BLOCKSIZE 32
#define UNROLL (4)

void do_block(int n, int si, int sj, int sk,
              double *A, double *B, double *C)
{
    for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 8)
    {
        for (int j = sj; j < sj + BLOCKSIZE; j++)
        {
            __m512d c[UNROLL];
            for (int r = 0; r < UNROLL; r++)
            {
                c[r] = _mm512_load_pd(C + i + r * 8 + j * n); //[ UNROLL];
                for (int k = sk; k < sk + BLOCKSIZE; k++)
                {
                    __m512d bb = _mm512_broadcastsd_pd(_mm_load_sd(B + j * n + k));
                    for (int r = 0; r < UNROLL; r++)
                        c[r] = _mm512_fmadd_pd(_mm512_load_pd(A + n * k + r * 8 + i), bb, c[r]);
                }
                for (int r = 0; r < UNROLL; r++)
                    _mm512_store_pd(C + i + r * 8 + j * n, c[r]);
            }
        }
    }
}

void dgemm(int n, double *A, double *B, double *C)
{
    for (int sj = 0; sj < n; sj += BLOCKSIZE)
        for (int si = 0; si < n; si += BLOCKSIZE)
            for (int sk = 0; sk < n; sk += BLOCKSIZE)
                do_block(n, si, sj, sk, A, B, C);
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