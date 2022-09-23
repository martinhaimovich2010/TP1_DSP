#%%
#Ejercicio 1

#Importo bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Defino los parametros de frecuencia, frecuencia de muestreo, numero de armonicos, etc
f0 = 440
fs = 44100
K = 5

#Defino la cantidad de muestras por 2 segundos (regla de 3)
N = (2*fs)//1

#Defino un vector n cuyos elementos irán desde 0 hasta el numero de muestra que se encuentra en el segundo 2
n = np.arange(N)

# Genero el vector tiempo de 2 segundos
t = np.linspace(0,2,N)

#Defino un vector A que esta lleno de ceros, cuya cantidad de elementos es igual a las muestras en 2 seg (N)
A = np.zeros(N)

#Voy agregandole elementos al vector A (armonicos) y grafico cada armonico
plt.figure(figsize=(25,15))
for k in range(1,K+1):
    armonicos = (1/k) * np.sin(2*n*k*np.pi*(f0/fs))
    A += armonicos
    plt.subplot(3,3,k)
    plt.plot(t,armonicos)
    plt.grid()
    plt.xlim(0,(8/f0))
    plt.ylim(-1,1)
    plt.title('Armónico %i' %k)
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

print("Se puede ver que la señal resultante es una diente de sierra")

#%%

#Ejercicio 2

import random

# Defino los parámetros
mu = 0
sigma = 1

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

print("Se puede ver que a medida que la señal aleatoria aumenta su dimensión, el desvío estandar tiende a 1, con un error cada vez menor, aproximandose al caso ideal de distribucion normal")

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
    return SNR

SNR1 = Señal_Ruido(AX1,sigma1,fs/f0)
SNR2 = Señal_Ruido(AX2,sigma2,fs/f0)
SNR3 = Señal_Ruido(AX3,sigma3,fs/f0)

#Agregandole a cada señal un componente de continua
C = int(input("Definir un valor para la componente de continua: "))

#Sumando el componente de continua a cada señal
AX1_C = AX1 + C
AX2_C = AX2 + C
AX3_C = AX3 + C

#Calculando SNR a cada señal con componente de continua
SNR4 = Señal_Ruido(AX1_C,sigma1,fs/f0)
SNR5 = Señal_Ruido(AX2_C,sigma2,fs/f0)
SNR6 = Señal_Ruido(AX3_C,sigma3,fs/f0)

# Creo los arrays de datos para la tabla
sdErrorList = np.array([("Señal 1", "Señal 2", "Señal 3"), (sigma1, sigma2, sigma3) , (SNR1, SNR2, SNR3), (SNR4, SNR5, SNR6)])
cell_text = sdErrorList.transpose()
colLabels = ['Señal con ruido', 'Sigma', 'SNR', 'SNR con continua']

# Grafico la tabla
plt.figure(figsize=(55,45))
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
plt.table(  cellText=cell_text,
            colLabels=colLabels,
            loc='center')
plt.show()

print("A medida que el desvio estandar es menor, la relacion señal ruido aumenta, este resultado es coherente con la formula planteada")
print("Si le sumo un componente de continua, si este valor es positivo, SNR aumenta") #NO SE SI ESTA BIEN ESTO

# FALTA: Analizar que pasa si agrego una componente de DC

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

# Creo los arrays de datos para la tabla
Average_SNR_List = np.array([("10", "100", "1000"), (averageA_RN10, averageA_RN100, averageA_RN1000), (SNR_average10, SNR_average100, SNR_average1000), (SNR1, SNR2, SNR3)])
cell_text = Average_SNR_List.transpose()
colLabels = ['Cantidad de señales de ruido', 'Promedio', 'SNR promedio', 'SNR ejercicio 3']

# Grafico la tabla
plt.figure(figsize=(25,15))
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
plt.table(  cellText=cell_text,
            colLabels=colLabels,
            loc='center')

# FALTA: Dar formato a tabla

# %%

#Ejercicio 5

from scipy import fftpack




def mediaMovilD(x, M):
    #Transformo la señal a filtrar y la normalizo
    X = np.fft.rfft(x) / np.amax(np.fft.rfft(x))
    print(X[0])
    #Defino vector de frecuencia
    f = fftpack.fftfreq(len(x))*fs
    #Grafico la señal a filtrar y su transformada
    plt.figure(1)
    plt.subplot(3,2,1)
    plt.plot(t, A)
    plt.xlim(0, 8/f0)
    plt.subplot(3,2,2)
    plt.plot(f[:-fs+1], abs(X))
    plt.xlim(0, 2500)
    plt.xticks([440, 880, 1320, 1760, 2200], ['440', '880','1320','1760','2200'])
    
    #Defino filtro de media movil y lo normalizo
    w = np.append(np.ones(M), np.zeros(len(t)-M))
    w = w / np.amax(w)
    #Grafico w
    plt.subplot(3,2,3)
    plt.plot(t, w)
    "plt.xlim(0, 8/f0)"
    #Transformo el filtro y lo normalizo
    W = np.fft.rfft(w) / np.amax(np.fft.rfft(w))
    #Grafico W, deberia quedar una funcion sinc
    """plt.subplot(3,2,4)
    plt.plot(f, W)
    plt.xlim(0, 2500)"""
    
    #Para filtrar, multiplico las transformadas de la señal y la del filtro
    #Es lo mismo que convolucionar la señal y la ventana
    Y = X * W
    #Grafico la señal filtrada
    plt.subplot(3,2,5)
    plt.plot(f[:-fs+1], abs(Y))
    plt.xlim(0, 2500)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud")
    plt.title("Señal transformada filtrada")
    
    #Antitransformo la señal filtrada
    y = np.fft.ifft(Y)
    #Grafico y en funcion del tiempo
    plt.subplot(3,2,6)
    plt.plot(np.linspace(0,2,len(y)), y)
    plt.xlim(0, 8/f0)
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.title("Señal original filtrada")
    
    return plt.plot(np.linspace(0,2,len(y)), y)

filtranding = mediaMovilD(A, 22000)

# NACHO -> Ver recursiva
#%%
#Ejercicio 6

M = 22000

w = 1/M * np.append(np.ones(M), np.zeros(len(t)-M))
h = np.convolve(A, w, mode='same')
h = h / np.amax(h)

plt.figure(1)
#Grafico la convolucion entre w y la señal del ejercicio 1
plt.subplot(1,2,1)
plt.plot(t, h)
plt.xlim(0, 8/f0)
plt.ylim(-1,1)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

#Grafico la señal filtrada del ejercicio 5
plt.subplot(1,2,2)
# filtranding = mediaMovilD(A, 1000)
plt.plot(t, y)
plt.xlim(0, 8/f0)
plt.ylim(-1,1)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.show()

"""def mediaMovilD(x, M):
    w = np.append(np.ones(M), np.zeros(len(t)-M))
    #Normalizo
    w = w / len(w)
    #Defino filtro de media movil
    w = (1/M) * np.ones(np.int(M))
    #Convoluciono el filtro con la señal
    x_conv = sig.fftconvolve(w, x, mode='same')
    return plt.plot(t, x_conv[:N]), plt.xlim(0, 8/f0)

filtrado1 = mediaMovilD(A, 10)"""

#%%

#Ejercicio 7

M = 100

a0 = 0.42
a1 = 0.5
a2 = 0.08
blackMan = np.zeros(M)
for i in range(M):
    blackMan[i] = a0 - a1 * np.cos((2*np.pi*i)/(M-1)) + a2 * np.cos((4*np.pi*i)/(M-1)) 

#Convoluciono el filtro black man con la señal del ejercicio 1
convBlack = np.convolve(A, blackMan, mode='same')
#normalizo
convBlack = convBlack / np.amax(convBlack)

#Grafico
plt.figure(1)
plt.plot(t, convBlack)
plt.xlim(0, 8/f0)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.show()

# %%
