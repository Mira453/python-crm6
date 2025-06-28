import numpy as np

# ----------------------------- ВХІДНІ ДАНІ -----------------------------

# Система 1: для методу Гауса
A1 = np.array([
    [1,  2, -1, -7],
    [8,  0, -9, -3],
    [2, -3,  7,  1],
    [1, -5, -6,  8]
], dtype=float)

b1 = np.array([-23, 39, -7, 30], dtype=float)

# Система 2: для методів Якобі та Зейделя
A2 = np.array([
    [14,  -4,  -2,   3],
    [-3,  23,  -6,  -9],
    [-7,  -8,  21,  -5],
    [-2,  -2,   8,  18]
], dtype=float)

b2 = np.array([38, -195, -27, 142], dtype=float)

# ----------------------------- МЕТОД ГАУСА -----------------------------

print("=== МЕТОД ГАУСА ===")
# Розв’язання системи
x1 = np.linalg.solve(A1, b1)
print("Розв’язок системи:")
print(x1)

# Визначник
detA1 = np.linalg.det(A1)
print("Визначник матриці A:", detA1)

# Обернена матриця
invA1 = np.linalg.inv(A1)
print("Обернена матриця A^-1:")
print(invA1)

# ----------------------------- МЕТОД ЯКОБІ -----------------------------

print("\n=== МЕТОД ЯКОБІ ===")
eps = 0.01
n = len(b2)
x_jacobi = np.zeros(n)
x_new_jacobi = np.zeros(n)
jacobi_iters = 0

for iteration in range(1000):
    for i in range(n):
        s = 0
        for j in range(n):
            if j != i:
                s += A2[i][j] * x_jacobi[j]
        x_new_jacobi[i] = (b2[i] - s) / A2[i][i]
    norm = np.linalg.norm(x_new_jacobi - x_jacobi, ord=np.inf)
    x_jacobi[:] = x_new_jacobi
    jacobi_iters += 1
    if norm < eps:
        break

print("Розв’язок:")
print(x_jacobi)
print("Кількість ітерацій:", jacobi_iters)

# ----------------------------- МЕТОД ЗЕЙДЕЛЯ -----------------------------

print("\n=== МЕТОД ЗЕЙДЕЛЯ ===")
x_seidel = np.zeros(n)
seidel_iters = 0

for iteration in range(1000):
    x_new_seidel = np.copy(x_seidel)
    for i in range(n):
        s1 = sum(A2[i][j] * x_new_seidel[j] for j in range(i))
        s2 = sum(A2[i][j] * x_seidel[j] for j in range(i + 1, n))
        x_new_seidel[i] = (b2[i] - s1 - s2) / A2[i][i]
    norm = np.linalg.norm(x_new_seidel - x_seidel, ord=np.inf)
    x_seidel[:] = x_new_seidel
    seidel_iters += 1
    if norm < eps:
        break

print("Розв’язок:")
print(x_seidel)
print("Кількість ітерацій:", seidel_iters)
