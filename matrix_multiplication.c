#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 512 // Valores : 32, 64, 128, 256, 512

void dgemm(int n, double *A, double *B, double *C)
{

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double cij = C[i + j * n]; /* cij = C[i][j] */
            for (int k = 0; k < n; k++)
                cij += A[i + k * n] * B[k + j * n]; /* cij += A[i][k]*B[k][j] */
            C[i + j * n] = cij;                     /* C[i][j] = cij */
        }
}

void buildMatrix(int n, double **P, double **Q, double **M)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            P[i][j] = rand();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Q[i][j] = rand();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = 0;
}

int main()
{
    double **MatrizP, **MatrizQ, **MatrizM;
    MatrizP =
        malloc(MATRIX_SIZE * sizeof(double *));
    MatrizQ =
        malloc(MATRIX_SIZE * sizeof(double *));
    MatrizM =
        malloc(MATRIX_SIZE * sizeof(double *));
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        MatrizP[i] =
            malloc(MATRIX_SIZE * sizeof(double));
        MatrizQ[i] =
            malloc(MATRIX_SIZE * sizeof(double));
        MatrizM[i] =
            malloc(MATRIX_SIZE * sizeof(double));
    }
    clock_t time;

    buildMatrix(MATRIX_SIZE, MatrizP, MatrizQ, MatrizM);
    time = clock();
    dgemm(MATRIX_SIZE, *MatrizP, *MatrizQ, *MatrizM);
    time = clock() - time;

    printf("O tempo de execução do programa é de: %0.6Lf segundos\n",
           ((long double)time) / CLOCKS_PER_SEC);
    return 0;
}