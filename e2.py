import modelo_lineal


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
costos = modelo.gradient_descent(0.1)
#modelo.imprimir()

# ejemplos
x = modelo.x
y = modelo.y

# curva de prediccion
x2 = [i-1 for i in range(10)]
y2 = map(lambda e: modelo.hipotesis(e), x2)

# costos por iteracion de descenso del gradiente
iteraciones = [i for i in range(len(costos))]

plt.plot(x, y, 'ro', label='ejemplos')
plt.plot(x2, y2, label='prediccion')
#plt.plot(iteraciones, costos)
plt.legend()
plt.show()

###############################################################################
# DATOS SOBRE MORTALIDAD (1.2)
###############################################################################
modelo = leer_archivo("data/x08.txt")
modelo.normalizar()
costos = modelo.gradient_descent(0.1)
#modelo.imprimir()

iteraciones = [i for i in range(len(costos))]
#plt.plot(iteraciones, costos)
#plt.show()
