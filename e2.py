import modelo_lineal
import numpy as np


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

    return modelo_lineal.Modelo(x, y, rasgos)

import matplotlib.pyplot as plt

###############################################################################
# DATOS SOBRE PESO (1.1)
###############################################################################
modelo = leer_archivo("data/x01.txt")
modelo.normalizar()
costos = modelo.gradient_descent(0.1, 100000)
#modelo.imprimir()

# costos por iteracion de descenso del gradiente
iteraciones = [i for i in range(len(costos))]

f1 = plt.figure(1)
plt.plot(iteraciones, costos, 'gx')
plt.title("Datos sobre peso\nCurva de convergencia de la funcion de costo")
plt.xlabel("Iteracion")
plt.ylabel("Costo")
plt.subplots_adjust(0.14)
plt.savefig("plots/e2-1a.png")

# ejemplos
x = modelo.x
y = modelo.y

# curva de prediccion
flatten = lambda l: [item for sublist in l for item in sublist]
x2 = np.arange(min(flatten(modelo.x)), max(flatten(modelo.x)), 0.5)
y2 = map(lambda e: modelo.hipotesis(e), x2)

f2 = plt.figure(num=2, figsize=(6, 4), dpi=180)
plt.plot(x, y, 'ro', label='ejemplos', markersize=1.3)
plt.plot(x2, y2, label='prediccion', linewidth=1)
plt.xlabel(modelo.rasgos[0])
plt.ylabel(modelo.rasgos[1])
plt.legend()
plt.subplots_adjust(0.14)
plt.savefig("plots/e2-1b.png")

#plt.show()


###############################################################################
# DATOS SOBRE MORTALIDAD (1.2)
###############################################################################
modelo = leer_archivo("data/x08.txt")
modelo.normalizar()
costos = modelo.gradient_descent(0.1)
#modelo.imprimir()

iteraciones = [i for i in range(len(costos))]

f3 = plt.figure(3)
plt.plot(iteraciones, costos, 'gx')
plt.title("Datos sobre mortalidad\nCurva de convergencia de la funcion de costo")
plt.xlabel("Iteracion")
plt.ylabel("Costo")
plt.subplots_adjust(0.14)
plt.savefig("plots/e2-2a.png")

#plt.show()
#raw_input()
