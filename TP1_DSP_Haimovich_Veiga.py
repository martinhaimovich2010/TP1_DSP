#%%
#Ejercicio 1
#Importo bibliotecas
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

# Genero el vector tiempo de 2 segundos
t=np.linspace(0,2,N)

#Defino un vector A que esta lleno de ceros, cuya cantidad de elementos es igual a las muestras en 2 seg (N)
A = np.zeros(N)

#Voy agregandole elementos al vector A (armonicos) y grafico cada armonico
for k in range(1,K+1):
    ck = ((-1)**k)/(k*np.pi)
    A += (1/k) * ck * np.sin(2*n*k*np.pi*(f0/fs))
    plt.subplot(3,3,k)
    plt.plot(t,A)
    plt.grid()
    plt.xlim(0,(16/f0))
    plt.title(("Señal LA 440 con ",k, "armonicos"))
    #plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.show()

#Normalizo la señal completa
Amax = np.amax(A)
A = A * (1/Amax)

#Grafico la señal completa
plt.figure(2)
plt.plot(t,A)
plt.grid()
plt.xlim(0,(16/f0))
plt.title('Señal LA 440')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud Normalizada")
plt.show()

#Grafico cada armónico por separado
#for k in range(1,K+1):
    #plt.subplot(3,3,k)
    #plt.plot(t,A)
    #plt.grid()
    #plt.xlim(0,(16/f0))
    #plt.title(("Señal LA 440 con ",k, "armonicos"))
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Amplitud Normalizada")
    #plt.
    #plt.show()

#%%

#Ejercicio 2

import random
#Defino una señal aleatoria de longitud L
L = int(input("Introducir longitud de la señal aleatoria: "))
valores = []
mu = 0
sigma = 1


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
    ds = (sumatoria/(len(valores)-1))**0.5
    error = 100-ds*100/1
    return "El desvio estandar obtenido es ", ds ," y el error es de ", error ,"%"
# %%

#Ejercicio 3

#Defino el desvio estandar de las 3 señales con ruido
sigma1 = 1
sigma2 = 0.1
sigma3 = 3

#Defino señales de ruido con desvio estandar 0.1, 1 y 3
x1 = np.random.normal(0, sigma1, len(t))
x2 = np.random.normal(0, sigma2, len(t))
x3 = np.random.normal(0, sigma3, len(t))

#Defino 3 señales con cada ruido respectivamente combinado a la señal del ejercicio 1
AX1 = A + x1
AX2 = A + x2
AX3 =  A + x3

#Normalizo las 3 señales nuevas
AX1max = np.amax(AX1)
AX1 = AX1 * (1/AX1max)
AX2max = np.amax(AX2)
AX2 = AX2 * (1/AX2max)
AX3max = np.amax(AX3)
AX3 = AX3 * (1/AX3max)

#Grafico las 3 señales normalizadas
plt.subplot(2,2,1)
plt.plot(t, AX1)
plt.xlabel("Tiempo")
plt.ylabel("AX1")
plt.xlim(0, 16/f0)
plt.show()

plt.subplot(2,2,2)
plt.plot(t, AX2)
plt.xlabel("Tiempo")
plt.ylabel("AX2")
plt.xlim(0, 16/f0)
plt.show()

plt.subplot(2,2,3)
plt.plot(t, AX3)
plt.xlabel("Tiempo")
plt.ylabel("AX3")
plt.xlim(0, 16/f0)
plt.show()

#Defino formula que calcula relacion señal ruido
def Señal_Ruido(A,sigma):
    SNR = A/sigma
    if SNR > 3:
        return "La relacion señal ruido es ", SNR
   


