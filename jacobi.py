import numpy as np
import multiprocessing as mp

def update_row(i, A, b, x):
    n = len(A)
    sigma = sum(A[i][j]*x[j] for j in range(n) if j != i)
    return (i, (b[i] - sigma) / A[i][i])

def jacobi_parallel(A, b, x_init, tol=1e-10, max_iter=5000):
    n = len(A)
    x = x_init.copy()

    for iteration in range(1,max_iter + 1):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(update_row, [(i, A, b, x) for i in range(n)])

        x_new = np.zeros_like(x)
        for i, value in results:
            x_new[i] = value

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Convergencia alcanzada en {iteration} iteraciones")
            return x_new
        
        x[:] = x_new

    print(f"Se alcanzo el maximo de {max_iter}iteracion sin converger ")
    return x
def verficar_solucion(A, b, x_sol):
    Ax = np.dot(A, x_sol)

    print("\nAx = np.dot(A, x)\n")
    print(Ax)
    print("\n")
    print(b)

    if np.allclose(Ax, b, atol=1e-10):
        print("La solución es correcta")
    else:
        print("La solución no es correcta")

if __name__=="__main__":
    A = np.array([[10, -1, 2, 0],
                  [-1, 11, -1, 3],
                  [2, -1, 10, -1],
                  [0, 3, -1, 8]], dtype=float)
    b = np.array([6, 25, -11, 15], dtype=float)
    x_init = np.zeros_like(b)

    x_sol = jacobi_parallel(A, b, x_init)
    print("\nResultado:", x_sol)

    verficar_solucion(A, b, x_sol)
