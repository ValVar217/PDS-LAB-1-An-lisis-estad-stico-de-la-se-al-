# PDS-LAB-1-Analisis-estadstico-de-la-señal-
Informe de laboratorio acerca del Análisis estadístico de una señal Biomedica. 

# INTRODUCCION
Mediante la realización de este laboratorio correspondiente a la asignatura de Procesamiento Digital de Señales (PDS), pudimos realizar un codigo utilizando Python (Spyder) para poder calcular y analizar datos estadísticos como la media, desviación estándar e histogramas de una señal Fisiologica obtenida de la base de datos PhysioNet. Además de ello, pudimos evaluar distintos tipos de ruido mediante la relación señal-ruido (SNR), todo lo anterior teniendo en cuenta la señal elegida (Neuropatía - EMG), con el objetivo de mejorar la interpretacion de dichas imagenes para un optimo diagnostico en el area de la salud. 

# ADQUISICION DE DATOS
# Señal (Emg) - Physio.Net:
Para llevar a cabo esta práctica, comenzamos buscando una señal electromiográfica (EMG) disponible en PhysioNet, pues la señal que elegimos corresponde a un pasiente con una Neuropatia. 
Para ser más especificos, la condicion del Paciente (Neuropatia/Neuropatia Periferica), basicamente consiste en un problema de los nervios que produce dolor, adormecimiento, cosquilleo, hinchazón y debilidad muscular en distintas partes del cuerpo. Esto por lo general, comienza en las manos o los pies y empeora con el tiempo. El cáncer o su tratamiento, como la quimioterapia, pueden causar neuropatía. También pueden causarla las lesiones físicas, las infecciones, las sustancias tóxicas o las afecciones como diabetes, insuficiencia de los riñones o desnutrición. Por otro lado, en una señal electromiográfica (EMG) de un paciente con esta condicion, podríamos observar varias alteraciones en la actividad muscular, tales como:  

1. Disminución de la amplitud
2. Aumento de la latencia: (Un retraso en la activación muscular causado por la conducción nerviosa más lenta o bloqueada).  
3. Actividad espontánea anormal
4. Disminución de la frecuencia de activación   
5. Alteraciones en la reclutación de unidades motoras: (Se pueden ver cambios en la cantidad y el tamaño de los potenciales de unidad motora, indicando reinnervación o pérdida neuronal). 

Ahora bien, luego de tener presente esta informacion, seguimos para descargar los archivos .dat y .hea correspondientes a dicha señal. Con estos archivos en nuestro poder, importamos la señal a Python (Spyder) y empleamos la librería wfdb para leer y visualizar los datos, facilitando así su analisis. 

![WhatsApp Image 2025-02-06 at 7 59 35 PM](https://github.com/user-attachments/assets/3cbfddf6-fe48-4e82-902b-c402e1e92217)  
|*Figura 1: Señal EMG de un paciente con neuropatía.*|

Pues de esta señal EMG correspondiente a una neuropatía, se pueden observar varias características relevantes como lo pueden ser:

1. Variaciones irregulares en la amplitud: Se aprecia una señal con picos positivos y negativos marcados, lo que podría indicar actividad anormal de las unidades motoras.  
2. Presencia de picos elevados y espaciados: Pueden representar descargas espontáneas o fibrilaciones, comunes en neuropatías debido a la disfunción nerviosa.  
3. Baja frecuencia en ciertas secciones: Hay momentos en los que la actividad es menos intensa, lo que puede reflejar una reducción en la activación muscular debido a la degeneración nerviosa.  
4. Patrón irregular en la señal: En una señal EMG normal, se espera una activación más uniforme y coordinada. Sin embargo, en esta imagen, la señal muestra patrones desorganizados que pueden deberse a una afectación en la conducción nerviosa.  

Por todo lo anterior, es escencial que podamos realizar las mediciones que se realizaran con el paso del codigo (Informe) para un optimo analisis por lo que tenemos lo siguiente:

# Importacion de la señal & Librerias:
El análisis como tal de los datos de la señal se realiza por medio de la programación mencionada, esto junto a librerías en donde tenemos a Numpy y SciPy para poder calcular nuestros estadísticos descriptivos, pues nuestro código comienza con la implementación de las librerías necesarias para el óptimo funcionamiento de nuestro proyecto: 

```python
import os ##ubicacion archivo
import wfdb #señal
import matplotlib.pyplot as plt #graficas
import numpy as np 
from scipy.stats import norm, gaussian_kde
import statistics   

os.chdir('C:/Users/lizav/Downloads')  #ubicacion
#print(os.getcwd())
datos, info = wfdb.rdsamp('emg_neuropathy', sampfrom=50, sampto=1000)
datos = np.array(datos).flatten()  # Convertir a 1D si es necesario
plt.figure(figsize=(10, 5))
plt.plot(datos, label="Señal EMG", color='c')
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")  
plt.title("Señal EMG Neuropatía")
plt.legend()
plt.grid() 
 ```
Pues esta sección es esencial ya que permite la manipulación del sistema de archivos y directorios, para leer y manipular registros de datos biomédicos almacenados que en este caso se encuentra en formato PhysioNet, para la generación de gráficos en este caso de la señal EMG en el dominio del tiempo, utilizando la librería de  matplotlib, se grafica la señal, asignando un color y una etiqueta incluyendo los ejes para indicar la unidad de tiempo y la amplitud (Caracteristicas estadísticas) utilizando herramientas de Python.
 
# Estadisticos Descriptivos:
Posteriormente, empezando con la medida de nuestros estadísticos descriptivos, que corresponden con la medición de la Media de la señal, la desviación estándar, el Coeficiente de variación, Histogramas y la Función de probabilidad: 

# 1.*La Media:*
   Se calcula la media de la señal de dos formas:
   
   a) Manualmente: Sumando todos los valores y dividiendo por el total de datos.  
   ![image](https://github.com/user-attachments/assets/8fc3d007-fdab-4428-ab9f-0abefbf4faa0)    
  |*Ecu 1: Formula para calcular la Media.*|    
   
   b) Usando Numpy.mean(), que proporciona un método más eficiente y rapido.
```python
#Media
sumatoriadatos = 0
for i in datos:
    sumatoriadatos += i
#datos sumados
media=sumatoriadatos/info['sig_len']
print(f"Media: {media}")

mean=np.mean(datos)
print(f"Media Numpy: {mean}")
```
	 
# 2.*Desviacion Estandar:*
   Se calcula la desviación estándar de la señal, que mide la dispersión de los valores respecto a 
   la media:

   a) Manualmente: Elevando al cuadrado la diferencia entre cada valor y la media, sumando y 
     dividiendo por (N-1).
     ![image](https://github.com/user-attachments/assets/0d53887a-40c8-41b2-8ebf-a1b651a7380e)    
     |*Ecu 2: Formula para calcular la Desviacion Estandar.*|    

   b) Usando numpy.std(), con ddof=1 para la corrección muestral.
```python
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
```
   
# 3.*Coeficiente de Variación:*
   El siguiente segmento de nuestro código, consiste en calcular el coeficiente de variación (CV) 
   que en este caso expresa la relación entre la desviación estándar y la media como un porcentaje, 
   es decir, Indica la variabilidad de la señal en comparación con su valor promedio:    
   ![image](https://github.com/user-attachments/assets/6180e8bf-87da-44c1-ad19-3100e5bc88e7)      
   |*Ecu 3: Formula para calcular el Coeficiente de Variación.*|      

```python
#COEFICIENTE DE VARIACIÓN
CV =(S/media)*100
print(f"Coeficiente de Variación: {CV}%")

# Calcular coeficiente de variación (en porcentaje)
cv = (desviacion_muestral / mean) * 100
print(f"Coeficiente de Variación Numpy: {cv:.2f}%")
```
# *Comparación de resultados:*  
Los resultados obtenidos manualmente y mediante **NumPy** muestran una gran similitud, con diferencias mínimas en la precisión de los decimales como se puede evidenciar acontinuación:  

![WhatsApp Image 2025-02-06 at 8 02 33 PM (1)](https://github.com/user-attachments/assets/0cc37e88-aed8-4c81-a6c9-ff589a3ede4e)   
|*Figura 2: Comparación de resultados (Manual vs NumPy) de los Estadisticos Descriptivos de La Media, Desviación Estandar y el Coeficiente de Variación.*|  

1. La **Media** calculada manualmente (=0.020559) y con NumPy (=0.020559) es prácticamente idéntica, con una diferencia en la última cifra decimal debido al redondeo automático de NumPy.      

2. En cuanto a la **Desviación Estándar**, el valor obtenido manualmente (=0.429913) tiene una mayor cantidad de decimales en comparación con el resultado de NumPy (=0.4299), por lo que podriamos decir que NumPy redondea el valor para facilitar su interpretación sin afectar significativamente la precisión de los resultados.  

3. Por ultimo, tenemos el **Coeficiente de Variación**, expresado en porcentaje, también presenta una leve diferencia en la cantidad de cifras decimales, pero el valor general es consistente entre ambos métodos:  Manual=2091.0712% y NumPy=2091.07%  

En conclusión, NumPy es una herramienta eficiente y precisa para el cálculo estadístico, ya que proporciona resultados rápidos con una precisión adecuada para la mayoría de las aplicaciones.  

# 4.*Histograma & Función de Probabilidad:* 
Este gráfico muestra la distribución de los valores de la señal EMG mediante un histograma y una estimación de densidad Kernel (KDE/permite visualizar de manera más clara la tendencia subyacente de la señal, evitando la dependencia de los límites de los bins del histograma), lo que proporciona una representación continua de la distribución de probabilidad de la amplitud de nuestra señal:

![WhatsApp Image 2025-02-06 at 7 59 53 PM](https://github.com/user-attachments/assets/ef88262a-5503-4c30-a268-3bec4370eb1d)  
|*Figura 3: Resultado de Histograma junto con la Función de Pobabilidad (Campana de Gauss).*|      

```python
#Histograma
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
```   
Del grafico de la *Figura 3* presentado muestra un **histograma** junto con la estimación de la **densidad de probabilidad (Campana de Gauss)** de una señal procesada digitalmente.    
  Pues de este grafico, podemos decir que (barras naranjas) representa la distribución de los valores de amplitud de la señal, evidenciando que la mayoría de los datos se concentran en torno a 0 (Cero), con una forma aproximadamente simétrica.  
Por otro lado, La curva azul representa una estimación de la distribución de probabilidad de los valores de la señal en donde se observa un pico pronunciado en torno a 0 (cero), lo que indica que la amplitud de la señal tiene una alta concentración en este valor, con menor probabilidad de valores alejados, lo que significa que esta en un Rango normal, es decir, es simetrica al rededor del eje en donde generalmente se encuentra la media.  

Es clave resaltar que se utiliza la herramienta graficadora plt. donde los datos de la señal son guardados por intervalos y renombrados "bins" en este caso se tienen 50 intervalos. Por otra parte se importa la biblioteca "gaussian_kde" para realizar una estimación del comportamiento de los intervalos bins y graficar una tendencia los más posibles cercanos (Función de probabilidad), como se muestra en la imagen
____________________________________________________________________________________________________

# RELACIÓN SEÑAL RUIDO (SNR):  
**Ruido en la Señal y Relación Señal-Ruido (SNR)**    
En nuestro contexto del procesamiento digital de señales, el **ruido** es cualquier perturbación no deseada que se superpone a la señal de interés, afectando su calidad y precisión. Estas interferencias pueden tener diversas fuentes, como componentes electrónicos, interferencias ambientales, artefactos fisiológicos (en señales biomédicas), o errores en la adquisición y transmisión de datos.    

 El ruido en una señal puede clasificarse en diferentes tipos, dependiendo de su origen y características, pues en este caso se utilizaron tres tipos de ruidos, los cueles son: 
 
- **Ruido Gaussiano**  
Es un tipo de ruido que sigue una distribución normal o gaussiana, caracterizada por una media de 0 y una desviación estándar que varía en función de la potencia del ruido. Se trata de un ruido aleatorio con propiedades estadísticas bien definidas, lo que lo hace ideal para simulaciones y análisis de sistemas en entornos controlados. Su presencia es común en dispositivos electrónicos y en la transmisión de señales debido a fenómenos térmicos y otras fuentes aleatorias de interferencia.  

- **Ruido de Impulso**  
Se caracteriza por la aparición de valores atípicos o picos de alta amplitud en momentos aleatorios dentro de la señal. Estos impulsos pueden ser positivos o negativos y tienen una duración muy breve en comparación con la señal original. Su impacto puede ser significativo, ya que introduce distorsiones abruptas que pueden alterar la interpretación de los datos. Es común en interferencias electromagnéticas, errores en la transmisión de datos y perturbaciones externas en sensores.  

- **Ruido de Artefacto**  
Se refiere a señales espurias o no deseadas que se introducen en un sistema de adquisición debido a factores externos. En el caso de señales biomédicas, este ruido puede originarse por movimientos del paciente, contracciones musculares involuntarias o fallas en los electrodos. En otros sistemas, puede deberse a interferencias mecánicas, acoplamientos eléctricos o problemas en los dispositivos de medición. Su presencia es particularmente problemática en análisis donde se requiere una alta precisión, ya que puede enmascarar información clave.  

# *Relación Señal-Ruido (SNR):*  
La SNR es una métrica que cuantifica la calidad de una señal en presencia de ruido. Se define como la relación entre la potencia de la señal útil y la potencia del ruido, expresada en decibeles (dB):  

![image](https://github.com/user-attachments/assets/1e8aea23-a961-4183-84d3-1365ea2d3079)  
|*Ecu. 4: Ecuacion para calcular el SNR en una señal Bimedica.*|  
____________________________________________________________________________________________________

Por lo anterior, tenemos la siguiente parte de nuestro codigo, en donde implmentamos la parte de la contaminacion de la señal con los 3 tipos de Ruidos mencionados con anterioridad por lo que:  

```python
# --- GENERAR RUIDO GAUSSIANO ---
ruido_std = np.std(datos) * 0.3  # 30% de la desviación estándar de la señal
ruido = np.random.normal(0, ruido_std, size=len(datos))  # Ruido gaussiano

# --- SEÑAL CONTAMINADA ---
señal_ruidosa = datos + ruido

# --- GRAFICAR SEÑALES ---
plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.plot(datos, color='c', label="Señal Original")
plt.title("Señal EMG Neuropatia ")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(señal_ruidosa, color='red', label="Señal con Ruido Gaussiano")
plt.title("Señal EMG Contaminada con Ruido Gaussiano")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
```  
Posteriormente tenemos el primer calculo del **SNR**, correspondiente a la señal **Gaussiana**, en donde se le agrego un **ruido Red**
```python
# --- CÁLCULO DEL SNR ---
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal_ruidosa**2)  # Potencia de la señal
    potencia_ruido = np.mean(ruido**2)  # Potencia del ruido
    snr = 10 * np.log10(potencia_señal / potencia_ruido)  # SNR en dB
    return snr

snr_original = calcular_snr(datos, np.zeros_like(datos))  # SNR sin ruido
snr_ruidoso = calcular_snr(datos, ruido)  # SNR con ruido

print(f"SNR con ruido Gaussiano: {snr_ruidoso:.2f} dB")
datos = datos.flatten()  # Convertir a una dimensión

# Frecuencia de muestreo (obtenida del archivo de información)
fs = info['fs']  # Frecuencia de muestreo en Hz

# Tiempo asociado a la señal
t = np.arange(0, len(datos)) / fs

# Parámetros del ruido de red
frecuencia_red = 60  # Frecuencia del ruido (60 Hz)
amplitud_ruido = 0.8  # Amplitud del ruido de red
ruido_red = amplitud_ruido * np.sin(2 * np.pi * frecuencia_red * t)

# Contaminación con ruido de red
datos_contaminados_red = datos + ruido_red
```   
El siguiente segmento de código, tiene la finalidad de generar y agregar **RUIDO DE PULSO** a una señal existente. Para ello, se define un rango de amplitud para los impulsos y se determina un porcentaje de la señal que será afectado por este ruido que se introduce de manera aleatoria. Este tipo de ruido se caracteriza por la aparición de **picos o valores extremos** en puntos específicos, lo que puede provocar distorsiones abruptas en la señal original:

```python
# Parámetros del ruido de pulso
amplitud_ruido_min = -2.5  # Valor mínimo del impulso
amplitud_ruido_max = 2.5   # Valor máximo del impulso
ruido_pulso = np.zeros_like(datos)
num_impulsos = int(0.05 * len(datos))  # 5% de la longitud total de la señal
indices_impulso = np.random.choice(len(datos), size=num_impulsos, replace=False)
ruido_pulso[indices_impulso] = np.random.uniform(amplitud_ruido_min, amplitud_ruido_max, size=num_impulsos)

# Contaminación con ruido de pulso
datos_contaminados_pulso = datos + ruido_pulso
```
Ahora bien, luego de ello Este código genera tres gráficas de señales electromiográficas (EMG) usando la librería `matplotlib` en Python, en un solo gráfico con tres subgráficas.

1. **Primera subgráfica** (señal EMG original)
2. **Segunda subgráfica** (señal EMG contaminada con ruido de red)
3. **Tercera subgráfica** (señal EMG contaminada con ruido de pulso):

En resumen, el código está generando un gráfico con tres subgráficas, cada una mostrando una señal EMG diferente: la original, una contaminada por **Ruido de Artefacto** y otra contaminada por **Ruido de Impulso** como se mostrara a continuacion: 

```python
# Gráficas
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t * 1000, datos, label="Señal EMG Original", color='c')
plt.title("Señal EMG Original")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t * 1000, datos_contaminados_red, label="Señal EMG Contaminada (60 Hz)", color='orange')
plt.title("Señal EMG Contaminada con Ruido de Red (60 Hz)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t * 1000, datos_contaminados_pulso, label="Señal EMG Contaminada con Ruido de Pulso", color='purple')
plt.title("Señal EMG Contaminada con Ruido de Pulso")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()
```

![WhatsApp Image 2025-02-06 at 8 01 20 PM](https://github.com/user-attachments/assets/a9fc06f2-006f-4df0-b9c4-3f888fe2d2ef)  
|*Figura 4: Señal con 2 contaminaciones 8Ruido Artefacto e Impulso) a la Señal original.*|  
En esta imagen (*Figura 4*) podemos decir que:  

1. **Señal EMG Original**: Presenta variaciones en la amplitud dentro de un rango reducido, con picos característicos de actividad muscular.
2. **Señal EMG Contaminada con Ruido de Red (60 Hz)**: Se observa una oscilación periódica superpuesta a la señal original (interferencia de 60 Hz).
3. **Señal EMG Contaminada con Ruido de Pulso**: Muestra variaciones abruptas e irregulares, probablemente causadas por artefactos de movimiento o interferencias externas.
____________________________________________________________________________________________________

El siguiente segmento de código, lo que hace es realizar varias operaciones de visualización y análisis de señales, incluyendo el cálculo de la relación señal-ruido (SNR) para dos tipos de ruido, así como la amplificación de una señal.

1. Visualización  
2. Cálculo del SNR para ruido de red
3. Cálculo del SNR para ruido de pulso  
4. Evaluación de amplitud

Osea, este código ajusta la visualización del gráfico, calcula y muestra la relación señal-ruido (SNR) para dos tipos de ruido, y amplifica la señal original mediante un factor de amplificación:

```python
plt.tight_layout()
plt.show()
# Cálculo del SNR para ruido de red
P_signal_red = np.mean(datos_contaminados_red ** 2)  # Potencia de la señal
P_noise_red = np.mean(ruido_red ** 2)  # Potencia del ruido de red
SNR_red = 10 * np.log10(P_signal_red / P_noise_red)
print(f"SNR con Ruido de Red: {SNR_red:.2f} dB")

# Cálculo del SNR para ruido de pulso
P_signal_pulso = np.mean(datos_contaminados_pulso ** 2)
P_noise_pulso = np.mean(ruido_pulso ** 2)
SNR_pulso = 10 * np.log10(P_signal_pulso / P_noise_pulso)
print(f"SNR con Ruido de Pulso: {SNR_pulso:.2f} dB")

###Evaluacion de amplitud
A = 2  # Factor de amplificación
datos_amplificados = A * datos  
```
Luego de esto, el código genera una figura con dos gráficos (subgráficos) utilizando **Matplotlib** de nuevo para visualizar nuestras señales como se muestra a continuación:  

1. Primer subgráfico (Señal EMG Amplificada)
2. Segundo subgráfico **(Señal EMG con Ruido Gaussiano)**  
3. Ajuste y visualización

```python
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(datos_amplificados, color='c', label="Señal Amplificada")
plt.title("Señal EMG Neuropatía (Amplificada)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(señal_ruidosa, color='red', label="Señal con Ruido Gaussiano")
plt.title("Señal EMG Contaminada con Ruido Gaussiano Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
```
![WhatsApp Image 2025-02-06 at 8 01 36 PM](https://github.com/user-attachments/assets/e74ebf6b-0457-46fd-a6bc-5ff2805be1b2)   
|*Figura 5: Señal con contaminación de Ruido Gaussiano) a la Señal original.*|    

En esta parte, la imagen presenta dos gráficos que muestran el comportamiento de una señal de electromiografía (EMG) en diferentes condiciones como por ejemplo: 

1. Gráfico superior: Representa la **señal EMG amplificada**. Pues en esta se observan variaciones en la amplitud con picos característicos de la actividad muscular.

2. Gráfico inferior: Muestra la misma **señal EMG, pero contaminada con ruido gaussiano**. Se evidencia un incremento en la variabilidad de la señal debido a la presencia del ruido, lo que dificulta la identificación de los picos originales.  
____________________________________________________________________________________________________
A continuacion, seguimos con el **Cálculo de la Relación Señal-Ruido (SNR)**  
Esta parte del código se encarga de calcular la SNR para evaluar la calidad de la señal de electromiografía (EMG) en presencia de distintos tipos de ruido.  

1. Definición de la función `calcular_snr'
2. Cálculo del SNR con ruido gaussiano:
3. Generación de señales contaminadas:    
4. Cálculo del SNR para ruido de red y ruido de pulso 

```python
# --- CÁLCULO DEL SNR ---
def calcular_snr(señal, ruido):
    potencia_señal = np.mean(señal**2)  # Potencia de la señal original
    potencia_ruido = np.mean(ruido**2)  # Potencia del ruido
    snr = 10 * np.log10(potencia_señal / potencia_ruido)  # SNR en dB
    return snr

snr_nuevo = calcular_snr(datos_amplificados, ruido)
print(f"SNR ruido Gaussiano con Amplitud Aplicada: {snr_nuevo:.2f} dB")

datos_con_red = datos_amplificados + ruido_red
datos_con_pulso = datos_amplificados + ruido_pulso

def calcular_snr(señal_original, ruido):
    P_signal = np.mean(señal_original ** 2)  # Potencia de la señal original
    P_noise = np.mean(ruido ** 2)  # Potencia del ruido
    return 10 * np.log10(P_signal / P_noise)

SNR_red = calcular_snr(datos_amplificados, ruido_red)
SNR_pulso = calcular_snr(datos_amplificados, ruido_pulso)  
```

Es importante tener en cuenta que esta es fundamental en aplicaciones como señales biomédicas (ECG, EMG, EEG), comunicaciones digitales y procesamiento de audio e imágenes.  Por consiguiente, tenemos la implementacion de la *Ecu. 4* para poder obtener la Relacion Señal-Ruido (SNR), en donde obtuvimos lo siguientes resultados:      

![WhatsApp Image 2025-02-06 at 8 03 05 PM](https://github.com/user-attachments/assets/f82d331b-1522-4f8c-9ddf-d754acd4f222)  
|*Figura 6: Resultados obtenidos mediante la implementacion de la ecuacion de SNR.*| 

Los resultados obtenidos para una señal EMG contaminada con distintos tipos de ruido, tanto en su forma original como después de aplicar una amplificación. Podemos comparan los resultados antes y después de la amplificación para evaluar su impacto en la calidad de la señal: 

1. **Ruido Gaussiano**:   
Presenta un **SNR más alto** en comparación con los otros ruidos, lo que indica que su efecto sobre la señal es menos dañina. Tambie, podemos tener en cuenta que el SNR aumenta con un **ruido = 11.04 dB** mientras que con **amplitud aplicada =16.80 dB** lo que quiere decir que aumento **5.76 dB**, por lo que podemos deducir que la amplificación mejora la relación señal-ruido de manera significativa.  

2. **Ruido de Red (60 Hz)**  
Tiene el **SNR más bajo** en ambas condiciones, lo que indica que la interferencia de la red  es **la más significativa** sobre la señal EMG. Podemos ver que la amplificación mejora el SNR ya que en el **Ruido = 1.85 dB** mientras que con **amplitud aplicada = 3.65 dB**,esto quiere decir que solo aumento **1.80 dB**, lo que sugiere que este tipo de ruido es difícil de mitigar.

3. **Ruido de Pulso**  
Su efecto fue **moderado**, en este caso podemos ver que la amplificación mejora el SNR igual que en los 2 casos anteriores, ya que en el **Ruido = 4.43 dB** mientras que con **amplitud aplicada = 8.21 dB**,esto quiere decir que solo aumento **3.78 dB**,lo que indica que es más efectivo en este caso que con el ruido de red.  
____________________________________________________________________________________________________

**Visualización y Análisis del Impacto del Ruido en la Señal EMG**  
En esta parte, se encarga de **graficar y analizar** una señal de electromiografía (EMG) en diferentes condiciones, permitiendo visualizar el efecto de distintos tipos de ruido sobre la señal original.  

1. Configuración de la Imagen
2. Primer Subgráfico: Señal EMG Amplificada
3. Segundo Subgráfico: Señal Contaminada con Ruido de Red (60 Hz) 
4. Tercer Subgráfico: Señal Contaminada con Ruido de Pulso
6. Cálculo e Impresión del SNR: 
     - SNR con Ruido de Red.  
     - SNR con Ruido de Pulso.  

```python
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t * 1000, datos_amplificados, label="Señal EMG Amplificada", color='c')
plt.title("Señal EMG Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t * 1000, datos_con_red, label="Señal con Ruido de Red (60 Hz)", color='orange')
plt.title("Señal EMG Contaminada con Ruido de Red Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t * 1000, datos_con_pulso, label="Señal con Ruido de Pulso", color='purple')
plt.title("Señal EMG Contaminada con Ruido de Pulso Amplificada")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print(f"SNR con Ruido de Red Amplitud aplicada: {SNR_red:.2f} dB")
print(f"SNR con Ruido de Pulso Amplitud Aplicada: {SNR_pulso:.2f} dB")
```  
Esta ultima parte tiene como fin, darle una nueva amplificacion/Ruido a la señal original para cada uno de los tipos de ruido desarrollados. Por lo que nuestra parte de programacion para esta instancia ya ha sido completado. 

____________________________________________________________________________________________________

# *Conclusión*
Esta práctica permitió aplicar técnicas de análisis y procesamiento de señales, destacando la importancia de mejorar la calidad de señales biomédicas en diagnósticos médicos. Los resultados obtenidos demuestran la eficacia del enfoque y su relevancia para escenarios con ruido en aplicaciones biomédicas reales.  

# *Librerias*
os (ubicacion archivo)  
wfdb (señal)  
matplotlib.pyplot as plt (graficas)  
numpy as np   
scipy.stats import norm, gaussian_kde  
statistics     

# *Bibliografia*  
[1] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, PC, Mark, R., ... y Stanley, HE (2000). PhysioBank, PhysioToolkit y PhysioNet: componentes de un nuevo recurso de investigación para señales fisiológicas complejas. Circulation [En línea]. 101 (23), págs. e215–e220.  

# *Licencia*  
DOI (versión 1.0.0): https://doi.org/10.13026/C24S3D  
Temas: neuropatía / electromiografía
