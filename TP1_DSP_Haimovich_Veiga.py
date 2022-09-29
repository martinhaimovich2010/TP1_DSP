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
import plotly.graph_objects as go

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

# Grafico en tabla
DC_SNR_layout = go.Layout(
    title='Desviación Estándar y Error según longitud de señal',
    margin=go.layout.Margin(
        autoexpand=True
    )
)

fig = go.Figure(data=[go.Table(header=dict(values=['L', 'Sigma', 'Error %'],align='center'),
                cells=dict(values=[np.array(arrayTabla).astype(str), np.around(dsArray, 2), np.around(errorArray,2)],align='center'))
                ],
                layout=DC_SNR_layout)
fig.show()

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
    SNR = np.round((np.amax(A)-np.mean(A))/sigma,2)
    return SNR

SNR1 = Señal_Ruido(AX1,sigma1,fs/f0)
SNR2 = Señal_Ruido(AX2,sigma2,fs/f0)
SNR3 = Señal_Ruido(AX3,sigma3,fs/f0)

#Agregandole a cada señal un componente de continua
DCcomps = [-10, 10, 1000]

#Sumando el componente de continua a cada señal
AX1_C1 = AX1 + DCcomps[0]
AX2_C1 = AX2 + DCcomps[0]
AX3_C1 = AX3 + DCcomps[0]

AX1_C2 = AX1 + DCcomps[1]
AX2_C2 = AX2 + DCcomps[1]
AX3_C2 = AX3 + DCcomps[1]

AX1_C3 = AX1 + DCcomps[2]
AX2_C3 = AX2 + DCcomps[2]
AX3_C3 = AX3 + DCcomps[2]

#Calculando SNR a cada señal con componente de continua
SNR4 = Señal_Ruido(AX1_C1,sigma1,fs/f0)
SNR5 = Señal_Ruido(AX2_C1,sigma2,fs/f0)
SNR6 = Señal_Ruido(AX3_C1,sigma3,fs/f0)

SNR7 = Señal_Ruido(AX1_C2,sigma1,fs/f0)
SNR8 = Señal_Ruido(AX2_C2,sigma2,fs/f0)
SNR9 = Señal_Ruido(AX3_C2,sigma3,fs/f0)

SNR10 = Señal_Ruido(AX1_C3,sigma1,fs/f0)
SNR11 = Señal_Ruido(AX2_C3,sigma2,fs/f0)
SNR12 = Señal_Ruido(AX3_C3,sigma3,fs/f0)

# Creo los arrays de datos para la tabla
sdErrorList = np.array([("Señal 1", "Señal 2", "Señal 3"), (sigma1, sigma2, sigma3) , (SNR1, SNR2, SNR3), (SNR4, SNR5, SNR6), (SNR7, SNR8, SNR9), (SNR10, SNR11, SNR12), (SNR13, SNR14, SNR15)])
cell_text = sdErrorList.transpose()
colLabels = ['Señal con ruido', 'Sigma', 'SNR', 'SNR con continua 1', 'SNR con continua 2', 'SNR con continua 3', 'SNR con continua 4']

DC_SNR_layout = go.Layout(
    title='Relaciones Señal-Ruido para distintas desviaciones estándar',
    margin=go.layout.Margin(
        autoexpand=True
    )
)

fig = go.Figure(data=[go.Table(header=dict(values=['Señal con ruido', 'Sigma', 'SNR', 'SNR con DC=-10', 'SNR con DC=10', 'SNR con DC=1000'],align='center'),
                cells=dict(values=[["Señal 1", "Señal 2", "Señal 3"], [sigma1, sigma2, sigma3] , [SNR1, SNR2, SNR3], [SNR4, SNR5, SNR6], [SNR7, SNR8, SNR9], [SNR10, SNR11, SNR12]],align='center'))
                ],
                layout=DC_SNR_layout)
fig.show()

print("A medida que el desvio estandar es menor, la relacion señal ruido aumenta, este resultado es coherente con la formula planteada.")
print("El SNR no se modifica con la adición de una componente de continua.") 

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

DC_SNR_layout = go.Layout(
    title='SNR para Promedio Ensamble',
    margin=go.layout.Margin(
        autoexpand=True
    )
)

fig = go.Figure(data=[go.Table(header=dict(values=['Cantidad de señales de ruido', 'SNR promedio', 'SNR ejercicio 3'],align='center'),
                cells=dict(values=[["10", "100", "1000"], [SNR_average10, SNR_average100, SNR_average1000], [SNR1, SNR2, SNR3]],align='center'))
                ],
                layout=DC_SNR_layout)
fig.show()


# %%

#Ejercicio 5

from scipy import fftpack

"""def mediaMovilD(x, M):
    L = x.shape[0]
    x_sample = np.zeros([L+(M-1), 1]) * 1.0
    x_sample[(M-1):] = x
    y = np.zeros(x.shape)
    for i in range(0, L):
        y[i] = x_sample[i, i+(M)].sum() / float(M)
    return y"""

"""def mediaMovilD(x, M):
    x_sample = x.shape[0]
    x_sample = x
    y = np.zeros(x.shape)
    for i in range(len(x)):
        y =+ x
    y = y/M
    y = y / np.amax(y)
    return y"""


"""def mediaMovilD(x, M):
    y = np.zeros(x.shape)
    for i in range(int(0 + M/2), int(len(t)-(M/2)+1)):
        y[i] = 0
        for j in range(int(-M/2), int((M/2)+1)):
            y[i] =+ x[i+j]
        y = y[i] / M
    return y"""

def mediaMovilD(x, M):
    t5 = x.shape[0]
    x_temp = np.zeros([t5, 1]) * 1.0
    x_temp = x
    y = np.zeros(x.shape)
    for i in range(0, t5):
        y[i] = x_temp[i:i+(M)].sum() / float(M)
    y = y / np.amax(y)
    return y


filtranding = mediaMovilD(A, 10)
plt.figure(1)
plt.plot(t, filtranding)
plt.xlim(0, 8/f0)
plt.ylim(-2,2)

def mediamovildr(x,M):
    if len(x)<M:
        raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
    if len(x)>M:
        y = np.zeros(len(x))
        acc=0.0
        for i in range(0,M//2):
            acc += x[i]
        y[M//2] = acc/M
        for i in range((M//2)+1,(len(y)-(M//2))):
            acc = acc + x[i+((M-1)//2)]-x[i-(((M-1)//2)+1)]
            y[i] = acc/M
        return (y-np.mean(y))/np.amax(y-np.mean(y)) # Esta normalización y desplazamiento deberían solucionarse de otra forma.
    else:
        s=len(x)-M
        return np.hstack([np.zeros(M-1),np.mean(x[s:s+M-1])])

xfr = mediamovildr(A,10)
print(np.amin(xfr))
plt.figure(2)
plt.plot(t, xfr)
plt.xlim(0, 8/f0)
plt.ylim(-2,2)


#%%
#Ejercicio 6

M = 22000

w = 1/M * np.append(np.ones(M), np.zeros(len(t)-M))
#Hago la convolucion entre la ventana y la señal
h = np.convolve(A, w, mode='full')
h = h / np.amax(h)

#Grafico la señal filtrada del ejercicio 5
plt.subplot(1,2,2)
filtranding = mediaMovilD(A, 1000)
# filtranding = mediaMovilD(A, 1000)
plt.plot(t, filtranding)
plt.xlim(0, 8/f0)
plt.ylim(-1,1)
    
plt.figure(1)

#Grafico la convolucion entre w y la señal del ejercicio 1
plt.subplot(1,2,1)
plt.plot(t, h[0:len(t)])
plt.xlim(0, 8/f0)
plt.ylim(-1,1)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.show()

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
convBlack = np.convolve(A, blackMan, mode='full')
#normalizo
convBlack = convBlack / np.amax(convBlack)

#Defino muestras iniciales y finales para graficar la señal filtrada quitando las muestras agregadas por la convolución en los extremos.
convBlackStart = ( (len(convBlack)-len(t)) // 2 )
convBlackEnd = ( (len(convBlack)+len(t)) // 2 )

#Grafico
plt.figure(1)
plt.plot(t, convBlack[0:len(t)])
plt.xlim(0.0031, 8/f0)
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.show()

# %%

#Ejercicio 8

#Importo libreria soundfile
import soundfile as sf
import sounddevice as sd

#Importo archivo de audio y lo guardo en variable respuesta al impulso h
h, fs = sf.read('Resp_Imp.wav')

#Defino funciones a utilizar
from conv_circular import circular_convolve, _periodic_summation

#Convolucion lineal
conv = np.convolve(A, h, mode='full')
conv = conv / np.amax(conv)

#Convolucion circular
#Usando funcion suma periodica igualo las longitudes de las 2 señales
in1 = _periodic_summation(A, len(h))
in2 = _periodic_summation(h, len(h))
#Convolucion
conv_circ = circular_convolve(A, h, len(h))
conv_circ = conv_circ / np.amax(conv_circ)

#Convolucion lineal mismo periodo que la circular
#Usando funcion suma periodica llevo las longitudes a la longitud de la convolucion lineal
in1 = _periodic_summation(A, len(conv))
in2 = _periodic_summation(h, len(conv))
#Convolucion
circ_lin = circular_convolve(A, h, len(conv))
circ_lin = circ_lin / np.amax(circ_lin)

#Grafico las señales convolucionadas

#Genero vectores de tiempo para cada señal a graficar
t1 = np.linspace(0,2,len(conv))
t2 = np.linspace(0,2,len(conv_circ))
t3 = np.linspace(0,2,len(circ_lin))

#Grafico convolucion lineal
plt.figure(figsize=(25,15))
plt.subplot(2,2,1)
plt.plot(t1, conv)
plt.xlim(0, 0.1)
plt.title("Convolucion lineal")

#Grafico convolucion circular
plt.subplot(2,2,2)
plt.plot(t2, conv_circ)
plt.xlim(0, 0.1)
plt.title("Convolucion circular")

#Grafico convolucion circular de misma longitud que la lineal
plt.subplot(2,2,3)
plt.plot(t3, circ_lin)
plt.xlim(0, 0.1)
plt.title("Conv circular (misma long que conv lineal)")

plt.show()

#Genero audios de las señales convolucionadas
sf.write('Conv.wav',conv,fs)
sf.write('Conv circular.wav',conv_circ,fs)
sf.write('Circular (igual lineal).wav',circ_lin,fs)


# %%

#Ejercicio 9

# Defino función signo con sgn(0)=1 , como se describe en la referencia.
def sgn(t):
    if t<0:
        return -1
    elif t>=0:
        return 1
    else:
        raise Exception('Invalid input. Input must be an integer or float.')

# Defino las funciones genéricamente.

def shortTimeEnergy(M,x,hop):
    if len(x)<M:
        raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
    ste = np.zeros(len(x)-M+1)
    w = np.hamming(M)
    for i in range(0,((len(x)-M)//hop)):
        for j in range(0,M-1):
            if (j+(i*hop)) < (len(x)-M+1):
                y = x[j+(i*hop)] * w[j]
                ste[j+(i*hop)] += ( ((y)**2) / M )   
    return ste

def zeroCrossingRate(M,x,hop):
    if len(x)<M:
        raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
    zcr = np.zeros(len(x)-M+1)
    w = np.hamming(M)
    for i in range(0,((len(x)-M)//hop)):
        for j in range(0,M-1):
            if (j+(i*hop)) < (len(x)-M+1):
                y = x[j+(i*hop)] * w[j]
                zcr[j+(i*hop)] += ( ( (sgn(y)) - (sgn(x[j+(i*hop)-1]*w[j])) ) / (2*M) )    
    return zcr

# Revisar!
def energyEntropy(M,x,K,hop):
    if len(x)<M:
        raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
    if M<K:
        raise Exception('La ventana K no debe tener más muestras que la ventana M')
    enen = np.zeros(len(x)-M+1)
    w = np.hamming(M)
    for i in range(0,((len(x)-M)//hop)):
        for j in range(0,M-1):
            if (j+(i*hop)) < (len(x)-M+1):
                y = x[j+(i*hop)] * w[j]
                eTotSF = ( np.sum(shortTimeEnergy(K, x[(i*hop):(i*hop)+K], hop//(M//K))) )
                if eTotSF > 0:
                    ej = ( (y**2) / (K) ) / eTotSF
                    enen[j+(i*hop)] += (-1) * ej * np.log2(ej)  
    return enen

# Importo las señales.
signal1, fs = sf.read('Sen_al1.wav')
signal2, fs = sf.read('Sen_al2.wav')
signal3, fs = sf.read('Sen_al3.wav')

# Calculo los parámetros temporales de las señales y grafico.
STE1 = shortTimeEnergy(1000, signal1, 500)
STE2 = shortTimeEnergy(1000, signal2, 500)
STE3 = shortTimeEnergy(1000, signal3, 500)

plt.figure(figsize=(25,15))
plt.subplot(1,3,1)
plt.plot(np.linspace(0,len(STE1)/fs,len(STE1)), np.array(STE1))
plt.subplot(1,3,2)
plt.plot(np.linspace(0,len(STE2)/fs,len(STE2)), np.array(STE2))
plt.subplot(1,3,3)
plt.plot(np.linspace(0,len(STE3)/fs,len(STE3)), np.array(STE3))
plt.show()

ZCR1 = zeroCrossingRate(1000, signal1, 500)
ZCR2 = zeroCrossingRate(1000, signal2, 500)
ZCR3 = zeroCrossingRate(1000, signal3, 500)

plt.figure(figsize=(25,15))
plt.subplot(1,3,1)
plt.plot(np.linspace(0,len(ZCR1)/fs,len(ZCR1)), np.array(ZCR1))
plt.subplot(1,3,2)
plt.plot(np.linspace(0,len(ZCR2)/fs,len(ZCR2)), np.array(ZCR2))
plt.subplot(1,3,3)
plt.plot(np.linspace(0,len(ZCR3)/fs,len(ZCR3)), np.array(ZCR3))
plt.show()

# ENEN not working yet.
ENEN1 = energyEntropy(1000, signal1, 200, 500)
ENEN2 = energyEntropy(1000, signal2, 200, 500)
ENEN3 = energyEntropy(1000, signal3, 200, 500)

plt.figure(figsize=(25,15))
plt.subplot(1,3,1)
plt.plot(np.linspace(0,len(ENEN1)/fs,len(ENEN1)), np.array(ENEN1))
plt.subplot(1,3,2)
plt.plot(np.linspace(0,len(ENEN2)/fs,len(ENEN2)), np.array(ENEN2))
plt.subplot(1,3,3)
plt.plot(np.linspace(0,len(ENEN3)/fs,len(ENEN3)), np.array(ENEN3))
plt.show()

# FALTA: Solucionar EnEn.

# %%

#Ejercicio 10

def spectralCentroid():
    return

def spectralFlux():
    return

def spectralRolloff():
    return

# %%

#Ejercicio 11



# %%

#Ejercicio 12



# %%

#Ejercicio 13


