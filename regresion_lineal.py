import numpy as np


class Modelo():
    def __init__(self, x, y, rasgos, theta=[]):
        self.x = x
        self.y = y
        self.rasgos = rasgos

        if not theta:
            self.theta = [1.0 for i in range(len(x[0]))]
        else:
            self.theta = theta

    def imprimir(self):
        for r in self.rasgos:
            print r,
        print
        for i in range(len(self.x)):
            print self.x[i], self.y[i]

    def normalizar(self):
        medias = np.mean(self.x, 0)  # medias de las columnas
        desv = np.std(self.x, 0)     # desviaciones estandar de las columnas

        self.x = self.x - medias
        self.x = self.x / desv

    def derivada(self, j):
        r = 0.0

        for i in range(len(self.x)):
            r += (np.dot(self.theta, self.x[i])-self.y[i])*self.x[i][j]

        return r/len(self.x)

    def gradient_descent(self, alpha=0.01, max_it=1000):
        temp = [0.0 for i in range(len(self.theta))]

        while (i < max_it):
            for j in range(len(self.x[0])):
                temp[j] = self.theta[j] - alpha*self.derivada(j)
            self.theta = temp[:]
            i += 1


def leer_archivo(filename):
    f = open(filename, 'r')

    while (True):
        line = f.readline()
        if line[0] != "#":
            break
        else:
            continue

    columnas = int(line.split()[0])

    line = f.readline()
    filas = int(line.split()[0])

    line = f.readline()
    rasgos = []
    for i in range(columnas-1):
        line = f.readline()
        rasgos.append(line)

    x = []
    y = []
    for i in range(filas):
        line = f.readline()
        split = line.split()
        x.append([])
        for j in range(columnas-2):
            x[i].append(float(split[j+1]))
        y.append(float(split[columnas-1]))

    return Modelo(x, y, rasgos)


modelo = leer_archivo("data/x01.txt")
modelo.normalizar()
modelo.gradient_descent()
