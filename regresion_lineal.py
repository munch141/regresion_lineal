import numpy as np


def leer_archivo(file):
    # falta esto
    return file


def derivada(x, y, theta, j):
    r = 0.0

    for i in range(len(x)):
        r += (np.dot(theta, x[i])-y[i])*x[i][j]

    return r/len(x)


def gradient_descent(x, y, theta, alpha=0.01, max_it=1000):
    temp = [0.0 for i in range(len(theta))]

    while (i < max_it):
        for j in range(len(x)):
            temp[j] = theta[j] - alpha*derivada(x, y, theta, j)
        theta = temp[:]
        i += 1

    return theta
