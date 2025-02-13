# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:14:30 2025

@author: lizav
"""

import os ##ubicacion archivo
import wfdb #señal
import matplotlib.pyplot as plt #graficas
import numpy as np 
from scipy.stats import norm, gaussian_kde
import statistics

os.chdir('C:\Users\marce\Documents\LAB - PROC. SEÑALES\LAB 1\Señal - Neuropatia EMG')  #ubicacion
#print(os.getcwd())
datos, info = wfdb.rdsamp('emg_neuropathy', sampfrom=50, sampto=1000)
plt.figure(figsize=(10, 5))
plt.plot(datos, label="Señal EMG", color='c')
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")  
plt.title("Señal EMG Neuropatía")
plt.legend()
plt.grid()
##grafica la señal

#Media
sumatoriadatos = 0
for i in datos:
    sumatoriadatos += i
#datos sumados
media=sumatoriadatos/info['sig_len']
print(f"Media: {media}")

mean=np.mean(datos)
print(f"Media Numpy: {mean}")



############
#DESVIACION ESTANDAR 
resta=datos-media
#print(resta)
resta2=resta**2
#print(resta2)
sumatoriaresta=0
for i in resta2:
    sumatoriaresta += i    
#print(sumatoriaresta)
S=np.sqrt(sumatoriaresta/(info['sig_len']-1)) ###nan
print(f"Desviacion estandar: {S}")

desviacion_muestral = np.std(datos, ddof=1)  # ddof=1 para muestra
print(f"Desviación estándar Numpy: {desviacion_muestral:.4f}")

############
#COEFICIENTE DE VARIACIÓN
CV =(S/media)*100
print(f"Coeficiente de Variación: {CV}%")

# Calcular coeficiente de variación (en porcentaje)
cv = (desviacion_muestral / mean) * 100
print(f"Coeficiente de Variación Numpy: {cv:.2f}%")
######


# Histograma
plt.figure()
plt.hist(datos, bins=50, edgecolor='black', alpha=1.0, color='orange', density=True)  # Normalizado para densidad
plt.grid()

# Estimación de la densidad mediante gaussian_kde
kde = gaussian_kde(datos.flatten())

# Ajustar los valores de KDE para que alcancen hasta 2.5 en el eje y
scaling_factor = 2.5 / max(kde(datos.flatten()))  # Factor de escalamiento
x_vals = np.linspace(datos.min(), datos.max(), 1000)
plt.plot(x_vals, kde(x_vals) * scaling_factor, color='blue', lw=2, label='Densidad KDE (escalada)')

# Limitar el eje Y
plt.ylim(0, 2.5)

plt.title("Histograma con Función de Probabilidad (KDE)")
plt.xlabel("Amplitud")
plt.ylabel("Densidad de Probabilidad")
plt.legend()
plt.show()


