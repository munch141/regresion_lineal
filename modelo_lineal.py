import numpy as np

class Modelo():
    def __init__(self, x, y, rasgos, theta=[]):
        self.x = x
        self.y = y
        self.rasgos = rasgos

        if not theta:
            self.theta = [1.0 for i in range(len(x[0])+1)]
        else:
            self.theta = theta

    def imprimir(self):
        print ("rasgos: ", self.rasgos)
        print ("theta: ", self.theta)
        print ("ejemplos:")
        for i in range(len(self.x)):
            print(self.x[i], self.y[i])

    def normalizar(self):
        medias = np.mean(self.x, 0)  # medias de las columnas
        desv = np.std(self.x, 0)     # desviaciones estandar de las columnas

        self.x = ((self.x - medias) / desv).tolist()

    def hipotesis(self, x):
        return self.theta[0]+np.dot(self.theta[1:], x)

    def derivada(self, j):
        r = 0.0

        if j == 0:
            for i in range(len(self.x)):
                r += (self.hipotesis(self.x[i])-self.y[i])
        else:
            for i in range(len(self.x)):
                r += (self.hipotesis(self.x[i])-self.y[i])*self.x[i][j-1]

        return r/len(self.x)

    def costo(self):
        r = 0.0
        for i in range(len(self.x)):
            r += (self.hipotesis(self.x[i])-self.y[i])**2

        return r/(2*len(self.x))

    def gradient_descent(self, alpha=0.01, max_it=1000):
        temp = [0.0 for i in range(len(self.theta))]
        J = []
        i = 0

        while (i < max_it):
            for j in range(len(self.theta)):
                temp[j] = self.theta[j] - alpha*self.derivada(j)
            self.theta = temp[:]
            J.append(self.costo())
            i += 1
        return J
