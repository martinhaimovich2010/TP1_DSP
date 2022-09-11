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
plt.figure(figsize=(25,15))
for k in range(1,K+1):
    ck = ((-1)**k)/(k*np.pi)
    ck_signal = (1/k) * ck * np.sin(2*n*k*np.pi*(f0/fs))
    A += ck_signal
    plt.subplot(3,3,k)
    plt.plot(t,ck_signal)
    plt.grid()
    plt.xlim(0,(8/f0))
    plt.title('Señal LA 440 con %i armonicos' %k)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")   

#Normalizo la señal completa
Amax = np.amax(A)
A = A * (1/Amax)

#Grafico la señal completa
plt.subplot(3,3,6)
plt.plot(t,A)
plt.grid()
plt.xlim(0,(8/f0))
plt.title('Señal LA 440 + Armónicos')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud Normalizada")
plt.show()

#Grafico cada armónico por separado
#plt.figure(figsize=(25,15))
#for k in range(1,K+1):
    #plt.subplot(3,3,k)
    #plt.plot(t,A)
    #plt.grid()
    #plt.xlim(0,(1/f0))
    #plt.title(("Señal LA 440 con ",k, "armonicos"))
    #plt.xlabel("Tiempo [s]")
    #plt.ylabel("Amplitud Normalizada")
#plt.show()

#%%

#Ejercicio 2

import random
# Defino una señal aleatoria de longitud L
# L = int(input("Introducir longitud de la señal aleatoria: "))
# valores = []

# Defino los parámetros
mu = 0
sigma = 1


#for i in range(L):
#    temp = random.gauss(mu, sigma)
#    valores.append(temp)

# Defino funcion para calcular el valor medio de un vector
def valor_medio(valores):
    sumatoria = 0
    for i in valores:
        sumatoria += i
    return sumatoria / len(valores)

# Defino funcion para calcular el desvío estandar de un vector
# medio = valor_medio(valores)
def desvio_estandar(medio, valores):
    sumatoria = 0
    for i in valores:
        sumatoria += (i - medio)**2
    ds = (sumatoria/(len(valores)-1))**0.5
    error = 100-ds*100/1
    # return "El desvio estandar obtenido es ", ds ," y el error es de ", error ,"%"
    return ds, error

# Inicializo arrays para guardar los datos
arrayTabla = [5, 10, 100, 1000, 10000, 100000]
averageArray = []
dsArray = []
errorArray = []

# Itero para dar valores a los arrays creados
for i in arrayTabla:
    L = i
    signal = []
    for i in range(L):
        temp = random.gauss(mu, sigma)
        signal.append(temp)
    average = valor_medio(signal)
    averageArray.append(average)
    ds, error = desvio_estandar(average, signal)
    dsArray.append(ds)
    errorArray.append(error)

# Creo los arrays de datos para la tabla
sdErrorList = np.array([np.array(arrayTabla).astype(str), np.around(dsArray, 2), np.around(errorArray,2)])
cell_text = sdErrorList.transpose()
colLabels = ['L', 'Sigma', 'Error %']

# Grafico la tabla
plt.figure(figsize=(25,15))
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
plt.table(  cellText=cell_text,
            colLabels=colLabels,
            loc='center')
plt.show()

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
plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
plt.plot(t, AX1)
plt.xlabel("Tiempo")
plt.ylabel("AX1")
plt.xlim(0, 16/f0)


plt.subplot(2,2,2)
plt.plot(t, AX2)
plt.xlabel("Tiempo")
plt.ylabel("AX2")
plt.xlim(0, 16/f0)


plt.subplot(2,2,3)
plt.plot(t, AX3)
plt.xlabel("Tiempo")
plt.ylabel("AX3")
plt.xlim(0, 16/f0)
plt.show()

#Defino formula que calcula relacion señal ruido
def Señal_Ruido(A,sigma,f_A):
    SNR = np.amax(A)/sigma
    # if SNR > 3:
        # return "La relacion señal ruido es ", SNR
    return SNR
    # else:
    #     signallessNoise = []
    #     for i in A:
    #         j=f_A
    #         if ((j % f_A) == 0):
    #             signallessNoise.append(i)
    #         j=j+1
    #     noisySNR, error = desvio_estandar(valor_medio(signallessNoise), signallessNoise)
    #     return noisySNR


SNR1 = Señal_Ruido(AX1,sigma1,fs/f0)
SNR2 = Señal_Ruido(AX2,sigma2,fs/f0)
SNR3 = Señal_Ruido(AX3,sigma3,fs/f0)
   
print(SNR1)
print(SNR2)
print(SNR3)

# FALTA: Presentar resultados en tabla y analizar que pasa si agrego una componente de DC

# %%

# Ejercicio 4

def promedio_ensamble(N):
    # Inicializo Array
    randNoiseSignals = []    

    # Creo las señales y las guardo en el array
    for i in range(N):
        A_randNoise = A + np.random.normal(0, 3, len(t))
        randNoiseSignals.append(A_randNoise)

    # Inicializo un nuevo array que será el promedio, y transpongo el array de señales con ruido para poder sumar los valores para el promedio
    averageA_RN = []
    randNoiseSignals_T = np.array(randNoiseSignals).transpose()
    for i in range(len(randNoiseSignals[0])):
        averageA_RN.append((1/10) * np.sum(randNoiseSignals_T[i]))

    # Mido SNR
    SNR_average = Señal_Ruido(averageA_RN,3,fs/f0)

    # Normalizo
    Amax = np.amax(averageA_RN)
    averageA_RN = averageA_RN / Amax

    return randNoiseSignals, averageA_RN, SNR_average

randNoiseSignals10, averageA_RN10, SNR_average10 = promedio_ensamble(10)
randNoiseSignals100, averageA_RN100, SNR_average100 = promedio_ensamble(100)
randNoiseSignals1000, averageA_RN1000, SNR_average1000 = promedio_ensamble(1000)

print(SNR_average10)
print(SNR_average100)
print(SNR_average1000)

plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
plt.plot(t, averageA_RN10)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud Normalizada")
plt.xlim(0, 16/f0)


plt.subplot(2,2,2)
plt.plot(t, averageA_RN100)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud Normalizada")
plt.xlim(0, 16/f0)


plt.subplot(2,2,3)
plt.plot(t, averageA_RN1000)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud Normalizada")
plt.xlim(0, 16/f0)
plt.show()

# FALTA: Presentar datos en tabla y chequear si el SNR está bien calculado (Ej. 3)

# %%