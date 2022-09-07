#Ejercicio 1
#Importo bibliotecas
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
# Defino los parametros de frecuencia, frecuencia de muestreo, numero de armonicos, etc
f0 = 440
fs = 44100
K = 5
#Defino la cantidad de muestras por 2 segundos (regla de 3)
N = (2*fs)//1
#Defino un vector n cuyos elementos irán desde 0 hasta el numero de muestra que se encuentra en el segundo 2
n = np.arange(N)
#Defino un vector A que esta lleno de ceros, cuya cantidad de elementos es igual a las muestras en 2 seg (N)
A = np.zeros(N)
for k in range(1,K+1):
    ck = ((-1)**k)/(k*np.pi)
    A += (1/k) * ck * np.sin(2*n*k*np.pi*(f0/fs))


#Normalizo la señal completa

# Genero el vector tiempo de 2 segundos
t=np.linspace(0,2,N)

#Grafico 
plt.figure(1)
plt.plot(t,A)
plt.grid()
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.show()


#%%

#Ejercicio 2

import random
#Defino una señal aleatoria de longitud L
L = int(input("Introducir longitud de la señal aleatoria: "))
valores = []

for i in range(L):
    temp = random.gauss(mu, sigma)
    valores.append(temp)

#Defino funcion para calcular el valor medio de un vector
def valor_medio(valores):
    sumatoria = 0
    for i in valores:
        sumatoria += i
    return sumatoria / len(valores)

#Defino funcion para calcular el desvío estandar de un vector
medio = valor_medio(valores)
def desvio_estandar(medio, valores):
    sumatoria = 0
    for i in valores:
        sumatoria += (i - medio)**2
    return (sumatoria/(len(valores)-1))**0.5