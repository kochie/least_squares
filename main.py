import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List

start: float = -100
end: float = 100
N: int = 100
r = 10

polynomial_coefficients: List[float] = [10, 0, 1, -2]

def create_data() -> np.ndarray:
    X = np.linspace(start, end, N)
    Y = np.random.rand(N)*r-r/2
    # for i, coeff in enumerate(polynomial_coefficients):
    #     Y = Y + np.power(X, len(polynomial_coefficients)-i-1)*coeff
    return np.array([X,Y])

def calcPolynomial(X: np.ndarray, polynomial_coefficients: List[float]) -> np.ndarray:
    Y = np.zeros(N)
    for i, coeff in enumerate(polynomial_coefficients):
        Y = Y + np.power(X, len(polynomial_coefficients)-i-1)*coeff
    return np.array(Y)

def least_square(data: np.ndarray, polynomial_length: int) -> np.ndarray:
    X = np.zeros((data.shape[1], polynomial_length), dtype=np.complex128)
    for i, x in enumerate(data[0]):
        for j in range(polynomial_length):
            X[i][j] = np.power(x, j)
    # print(X)
    Xt = np.transpose(X)
    # print(Xt@X)
    a = np.linalg.inv(Xt@X)@Xt@data[1]
    # print(a)
    return a

def calcResidual(Y1: np.ndarray, Y2: np.ndarray) -> float:
    return (np.square(Y1 - Y2)).mean()

if __name__ == "__main__":
    data = create_data()
    a = least_square(data, 80)
    print(a)

    # y1 = calcPolynomial(data[0], polynomial_coefficients)
    # print(calcResidual(data[1], y1))
    y2 = calcPolynomial(data[0], a[::-1])
    print(calcResidual(data[1], y2))

    fig, ax = plt.subplots()
    ax.scatter(data[0], data[1], label='input data')
    # ax.plot(data[0], y1, label='original line')
    ax.plot(data[0], y2, label='regression line')
    ax.legend()

    ax.set(xlabel='X', ylabel='Y', title='About as simple as it gets, folks')
    ax.grid()

    # fig.savefig("test.png")
    plt.show()

