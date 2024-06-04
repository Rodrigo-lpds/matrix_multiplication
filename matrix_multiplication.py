# execute to run: python3 matrix_multiplication.py

from decimal import Decimal
import timeit
import random

begins = timeit.default_timer()

def multiply(m_0,m_1):
    n = len(m_0)
    result = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += m_0[i][k] * m_1[k][j]
    
    return result

def build(n_lines, n_cols):
    matrix = []
    linha = []
    while len(matrix) != n_lines:
        n = Decimal(random.randint(0, 99))
        linha.append(n)
        if len(linha) == n_cols:
            matrix.append(linha)
            linha = []
    return matrix

P = []
Q = []

q = 2048 #Values: 128, 256, 512, 1024, 2048

P = build(q, q)
Q = build(q, q)
result = multiply(P, Q)

end_progr = timeit.default_timer()
print('O tempo de execução do programa para uma matrix %i x %i foi de: %f segundos ' % (q,q,end_progr -
begins))