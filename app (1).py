import streamlit as st
#!/usr/bin/env python
# coding: utf-8

st.write("""
# <center> <h1>Universidad Nacional de San Agustín de Arequipa</h1> </center>
#
#<center> <h1>Escuela Profesional de Ingeniería de Telecomunicaciones</h1> </center>
# 
# <center> <h1> </h1> </center> 
# <center><img src="https://www.unsa.edu.pe/wp-content/uploads/sites/3/2018/05/Logo-UNSA.png" width="380" height="4200"></center>
# 
# <center> <h2>Ingeniero Renzo Bolivar - Docente DAIE</h2> </center>

# <center> <h1>Curso : Computación 1</h1> </center> 

# ![linea 1](https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png)

# <center> <h2>GRUPO A - Nº1</h2> </center> 
# <h2>Alumnos:  </h2>
# <h2>    
#     - Adolfo Carpio Beato
#     - Elmer Flores Mallma
#     - Melany Guizado Ttito
#     - Molly Caceres Ramos
#     - Gustavo Zapana Callocondo
#
# </h2>
# 

# ![linea 1](https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png)

# <center> <h1>INVESTIGACIÓN FORMATIVA</h1> </center> 
# <center> <h1>PROYECTO FINAL</h1> </center> 
# <center> <h1>PYTHON - Inteligencia Artificial</h1> </center> 

# ![linea 1](https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png)

# ## OBJETIVOS

# Los Objetivos de la investigación formativa son:
# 
# - **Competencia Comunicativa** Presentación de sus resultados con lenguaje de programación Python utilizando los archivos Jupyter Notebook.
# - **Competencia Aprendizaje**: con las aptitudes en **Descomposición** (desarticular el problema en pequeñas series de soluciones), **Reconocimiento de Patrones** (encontrar simulitud al momento de resolver problemas), **Abstracción** (omitir información relevante), **Algoritmos** (pasos para resolución de un problema).
# - **Competencia de Trabajo en Equipo**: exige habilidades individuales y grupales orientadas a la cooperación, planificación, coordinación, asignación de tareas, cumplimiento de tareas y solución de conflictos en pro de un trabajo colectivo, utilizando los archivos Jupyter Notebook los cuales se sincronizan en el servidor Gitlab con comandos Git.

# ![linea 1](https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png)

# <center> <h1>Aplicación en IA</h1> </center> 
# <center> <h1>Sistema Recomendador</h1> </center> 

# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# <div class="alert alert-info">
# 
#     
#    La <strong>compatibilidad o similitud</strong> será encontrada con el algoritmo de <strong>Correlación de Pearson</strong> y será verificada con la <strong>La Matrix de Correlación de Pearson con una librería de Python y utilizando una función personal</strong>
#     
# </div>

# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# <center> <h1>Base Teórica</h1> </center> 

# ## Análisis de Correlación

# El **análisis de correlación** es el primer paso para construir modelos explicativos y predictivos más complejos.

# <div class="alert alert-info">
# 
#    A menudo nos interesa observar y medir la <strong>relación entre 2 variables numéricas</strong> mediante el análisis de correlación. 
#    Se trata de una de las *técnicas más habituales en análisis de datos* y el primer paso necesario antes de construir cualquier <strong>modelo explicativo o predictivo más complejo</strong>.
#    Para poder tener el  Datset hay que recolectar información a travez de encuentas.
#     
# </div>

# Conocer los datos y la tipología de las variables: bien de tipo cuantitativo o cualitativo. Dado que en función del tipo de las variables del conjunto de datos, se aplicarán una serie de coeficientes u otros.
# 
# ### Método para su aplicación
# 
# Conocer qué método es el más adecuado para su aplicación: de nada vale conocer el script que ejecuta una correlación, si no se puede basar matemáticamente cómo desarrollar los cálculos.
# 
# Conocer, comprender e interpretar los resultados que los indicadores ofrecen: parte fundamental del análisis, basada en la diferencia existente entre los conceptos de causalidad y correlación que subyace en el análisis, y que suelen confundirse. El mero cálculo de un coeficiente, no determinado por una causalidad, no tiene porqué ser válido. Es decir «la existencia de correlación, no implica causalidad», como se refleja en esta viñeta:
# 
# Lo primero, un «conjunto de datos».
# En dicho conjunto el requisito es tener mínimo dos «variables» (éste es el nombre técnico), según unas técnicas u otras se han denominado «dominios», «campos» de una base de datos y similares.
# Conocer las bases matemáticas y estadísticas del análisis de datos, en concreto:

# ### ¿Qué es la correlación?

# La correlación es un tipo de asociación entre dos variables numéricas, específicamente evalúa la **tendencia (creciente o decreciente) en los datos**.
# 
# Dos variables están asociadas cuando una variable nos da información acerca de la otra. Por el contrario, cuando no existe asociación, el aumento o disminución de una variable no nos dice nada sobre el comportamiento de la otra variable.
# 
# Dos variables ***se correlacionan cuando muestran una tendencia creciente o decreciente***.

# ### Correlación simple

# En este apartado nos vamos a centrar en el estudio de un tipo particular de relación llamada lineal y nos vamos limitar a considerar únicamente dos variables (simple).
# Una relación lineal positiva entre dos variables X e Y indica que los valores de las dos variables varían de forma parecida: los sujetos que puntúan alto en X tienden a puntuar alto en Y y los que puntúan bajo en X tienden a puntuar bajo en Y. Una relación lineal negativa significa que los valores de las dos variables varían justamente al revés: los sujetos que puntúan alto en X tienden a puntuar bajo en Y y los que puntúan bajo en X tienden a puntuar alto en Y.
# La forma más directa e intuitiva de formarnos una primera impresión sobre el tipo de relación existente entre dos variables es a través de un **diagrama de dispersión**. 
# 
# Un diagrama de dispersión es un gráfico en el que una de las variables (X) se coloca en el eje de abscisas, la otra (Y) en el de ordenadas y los pares (x , y) se representan como una nube de puntos. La forma de la nube de puntos nos informa sobre el tipo de relación existente entre las variables. 

# Un diagrama de dispersión nos permite formarnos una idea bastante aproximada sobre el tipo de relación existente entre dos variables, también puede utilizarse como una forma de cuantificar el grado de relación lineal existente entre dos variables: basta con observar el grado en el que la nube de puntos se ajusta a una línea recta.

# Sin embargo, utilizar un diagrama de dispersión como una forma de cuantificar la relación entre dos variables no es, en la práctica, tan útil como puede parecer a primera vista. Esto es debido a que la relación entre dos variables no siempre es perfecta o nula: habitualmente no es ni lo uno ni lo otro. Podríamos encontrar una línea recta ascendente que representara de forma bastante aproximada el conjunto total de los puntos del diagrama.

# ### Correlación parcial

# El procedimiento Correlaciones parciales permite estudiar la relación lineal existente entre dos variables controlando el posible efecto de una o más variables extrañas. Un coeficiente de correlación parcial es una técnica de control estadístico que expresa el grado de relación lineal existente entre dos variables tras eliminar de ambas el efecto atribuible a terceras variables.
# 
# Por ejemplo, se sabe que la correlación entre las variables inteligencia y rendimiento escolar es alta y positiva. Sin embargo, cuando se controla el efecto de terceras variables como el número de horas de estudio o el nivel educativo de los padres, la correlación entre inteligencia y rendimiento desciende, lo cual indica que la relación entre inteligencia y rendimiento está condicionada, depende o está modulada por las variables sometidas a control.

# ### ¿Cómo se mide la correlación?

# Tenemos el coeficiente de **correlación lineal de Pearson** que se *sirve para cuantificar tendencias lineales*, y el **coeficiente de correlación de Spearman** que se utiliza para *tendencias de aumento o disminución, no necesariamente lineales pero sí monótonas*. 

# ### Correlación de Pearson

# <div class="alert alert-info">
# El coeficiente de correlación lineal de Pearson mide una tendencia lineal entre dos variables numéricas.
# </div>

# Es el método de correlación más utilizado, pero asume que:
# 
#  - La tendencia debe ser de tipo lineal.
#  - No existen valores atípicos (outliers).
#  - Las variables deben ser numéricas.
#  - Tenemos suficientes datos (algunos autores recomiendan tener más de 30 puntos u observaciones).
# 
# Los dos primeros supuestos se pueden evaluar simplemente con un diagrama de dispersión, mientras que para los últimos basta con mirar los datos y evaluar el diseño que tenemos.

# En estadísticas , el coeficiente de correlación de Pearson también referido como de Pearson r , el Pearson coeficiente de correlación momento-producto ( PPMCC ), o la correlación bivariada, es una medida de correlación lineal entre dos conjuntos de datos. Es la covarianza de dos variables, dividida por el producto de sus desviaciones estándar.; por lo tanto, es esencialmente una medida normalizada de la covarianza, de modo que el resultado siempre tiene un valor entre -1 y 1. Al igual que con la covarianza en sí, la medida solo puede reflejar una correlación lineal de variables e ignora muchos otros tipos de relación o correlación. Como ejemplo simple, uno esperaría que la edad y la altura de una muestra de adolescentes de una escuela secundaria tuvieran un coeficiente de correlación de Pearson significativamente mayor que 0, pero menor que 1 (ya que 1 representaría una correlación irrealmente perfecta).

# ## Interpretación del tamaño de una correlación

# ### Inferencia
# La inferencia estadística basada en el coeficiente de correlación de Pearson a menudo se centra en uno de los dos objetivos siguientes:
# 
# Un objetivo es probar la hipótesis nula de que el verdadero coeficiente de correlación ρ es igual a 0, basado en el valor del coeficiente de correlación muestral r .
# El otro objetivo es derivar un intervalo de confianza que, en un muestreo repetido, tenga una probabilidad determinada de contener ρ .

# ### Usando la transformación de Fisher
# En la práctica, los intervalos de confianza y las pruebas de hipótesis relativas a ρ se suelen realizar mediante la transformación de Fisher ,F.

# ### Usando un bootstrap
# 
# El bootstrap se puede utilizar para construir intervalos de confianza para el coeficiente de correlación de Pearson. En el bootstrap "no paramétrico", n pares ( x i ,  y i ) se muestrean "con reemplazo" del conjunto observado de n pares, y el coeficiente de correlación r se calcula basándose en los datos remuestreados. Este proceso se repite un gran número de veces y la distribución empírica de los valores r remuestreados se utilizan para aproximar la distribución muestral del estadístico. Un intervalo de confianza del 95% para ρ se puede definir como el intervalo que abarca desde el percentil 2.5 al 97.5 de los valores r remuestreados .

# ### Gráfica de dispersión

# Es un tipo de ```diagrama``` que se utiliza para __observar y analizar__ la existencia de una __correlación__ entre ```dos variables``` cuantitativas, donde una depende de la otra mediante una relación determinada para verificar o comprobar una hipótesis. Brinda una visualización de ```datos compacta```, que la hace apropiada para estudiar resultados cuantiosos de __encuestas y pruebas__.
# 
# Como su nombre indica, consiste en una ```colección de puntos``` esparcidos en el plano cartesiano XY, donde se desea distinguir un __patrón o comportamiento__ en el conjunto de datos para comprender su ```distribución```.

# ### Cómo se interpreta la correlación

# El signo nos indica la dirección de la relación, como hemos visto en el diagrama de dispersión.
#  - un valor positivo indica una relación directa o positiva,
#  - un valor negativo indica relación indirecta, inversa o negativa,
#  - un valor nulo indica que no existe una tendencia entre ambas variables (puede ocurrir que no exista relación o que la relación sea más compleja que una tendencia, por ejemplo, una relación en forma de U).

# La magnitud nos indica la fuerza de la relación, y toma valores entre $-1$ a $1$. Cuanto más cercano sea el valor a los extremos del intervalo ($1$ o $-1$) más fuerte será la tendencia de las variables, o será menor la dispersión que existe en los puntos alrededor de dicha tendencia. Cuanto más cerca del cero esté el coeficiente de correlación, más débil será la tendencia, es decir, habrá más dispersión en la nube de puntos.
#  - si la correlación vale $1$ o $-1$ diremos que la correlación es “perfecta”,
#  - si la correlación vale $0$ diremos que las variables no están correlacionadas.

# 
# <center><img src="https://user-images.githubusercontent.com/25250496/204172549-2ccf3be3-a2b3-4b49-9cd4-adb66e28621d.png" width="700" height="4200"></center>
# 
# 
# 

# <center> <h3>Fórmula Coeficiente de Correlación de Pearson</h3> </center>  
# <center> <h3> </h3> </center> 
# $$ r(x,y)=\frac{\sum_{i=1}^{n}(x_{i}-\overline{x})(y_{i}-\overline{y})}{\sqrt{\sum_{i=1}^{n}(x_{i}-\overline{x})^{2}}\sqrt{\sum_{i=1}^{n}(y_{i}-\overline{y})^{2}}}$$

# **Distancia Euclidiana**: La distancia euclidiana es la generalización del __`teorema de Pitágoras`__.

# $$d_{E}(x,y)=\sqrt{\sum_{i=1}^{n}(x_{i}-y_{i})^{2}}$$

# ### Correlación de Spearman

# <div class="alert alert-info">
#     El coeficiente de correlación lineal de Spearman mide la fuerza y la dirección de la asociación entre dos variables clasificadas.
# </div>

# Se emplea como alternativa no paramétrica al coeficiente de Pearson cuando los valores son ordinales, o bien, cuando los valores son continuos pero no satisfacen la condición de normalidad.

# $$ r(s)=1-\frac{6\sum {di}^{2}}{n(n²-1)}$$

# Siendo _di_ la la distancia entre los rangos de cada observación (*_xi - yi_*) y n el número de observaciones.
# 

# El coeficiente de Spearman requiere que la relación entre las variables sea monótona, es decir, que cuando una variable crece la otra también lo hace o cuando una crece la otra decrece

# ### Correlación de Kendall

# <div class="alert alert-info">
#     El coeficiente de correlación lineal de Kendall mide la intensidad de una relación lineal entreb dos variables, que no pudieran no tener relación causal entre si, y sin embargo estar relaciodas.
# </div>

# Es otra alternativa al coeficiente de correlación de Pearson cuando no se cumple la condición de normalidad. Se utiliza cuando el número de observaciones es pequeño o los valores se acumulan en una región por lo que el número de ligaduras al generar el ranking es alto.

# $$ t =\frac{C-D}{\frac n2 (n-1)}$$

# siendo  *C*  el número de pares concordantes, aquellos en los que el rango de la segunda variable es mayor que el rango de la primera variable.  *D*  el número de pares discordantes, cuando el rango de la segunda es igual o menor que el rango de la primera variable.

# ### Coeficientes de correlación lineal
# Son estadísticos que cuantifican la asociación lineal entre dos variables numéricas. Existen diferentes tipos, de entre los que destacan el Pearson, Spearman y Kendall. Todos ellos comparten que:
# - Su valor está comprendido en el rango [+1 , -1]. Siendo +1 una correlación positiva perfecta y -1 una correlación negativa perfecta.
# - Se emplean como medida de la fuerza de asociación entre dos variables (tamaño del efecto):

# - **0** :asociación nula
# 
# - **0.1** : asociación pequeña.
# 
# - **0.3**: asociación mediana.
# 
# - **0.5**: asociación moderada.
# 
# - **0.7**: asociación alta.
# 
# - **0.9**: asociación muy alta.

# Desde el punto de vista práctico, las principales diferencias entre estos tres coeficientes son:
# - La correlación de Pearson funciona bien con variables cuantitativas que tienen una distribución normal o próxima a la normal.
# - La correlación de Spearman se emplea con variables cuantitativas.En lugar de utilizar directamente el valor de cada variable, los datos son ordenados y reemplazados por su respectivo orden ranking.
# - Al igual que la correlación de Spearman, utiliza la ordenación de las observaciones ranking. Es recomendable cuando se dispone de pocos datos y muchos de ellos ocupan la misma posición en el rango.

# #### Covarianza
# Mide el grado de variación conjunta de dos variables aleatorias.

# $$ Cov(X,Y)=\frac{\sum_{i=1}^{n}(x_{i}-\overline{x})(y_{i}-\overline{y})}{N-1}$$

# donde x̄ e  ̅y son la media de cada variable, y _xi_ e _yi_ son el valor de las variables para la observación _i_.
# Los valores positivos indican que las dos variables cambian en la misma dirección y los valores negativos en direcciones opuestas.

# ## ¿Qué son los datos faltantes?

# Cuando se realiza una encuesta es probable que algunas respuestas no esten constestadas debido a diversos factores como: no entender la pregunta realizada en la encuesta, el rechazo de la persona a lo hora de informar un tema que no quiere que las demas personas sepan, el cansancio de responder muchas preguntas, etc.
# 
# 

# __`Una técnica muy conocida para el procedimiento de los faltantes es la imputación`__. 

# ### ¿Qué es la imputación?
# Es llenar los espacios vacios de la base de datos imcompleta con valores con valores admisibles y así obtener un archivo completo para poder analizarlo. Se habla de la imputación como un proceso de `fabricación de datos`. La imputación nos ayuda a:
# - Facilitar los procesos posteriores de análisis de datos.
# - Facilitar la consistencia de los resultados entre distintos tipos de análisis.
# 

# Las técnicas de imputación se clasifican en:

# ### Procedimientos tradicionales de imputación

# #### Análisis con datos complejos:
# El ***listwise*** es el método que se utiliza con mayor frecuencia, consta de una manera de proceder en donde se trabaja únicamente con las observaciones que contienen la información completa.
# 
# #### Análisis con los datos disponibles:
# El procedimiento ***Available-case (AC)***; en comparación con el anterior este utiliza distintos tamaños de muestra; este método asume un patrón MCAR en los datos obtenidos; las observaciones que no tienen datos se eliminan.
# 
# ####  Reponderación:
# Los ponderadores se interpretan como el número de unidades de la población que representa a cada elemento de muestra; los algoritmos completaran con estimaciones compatibles. 

# ### Imputación simple

# #### - Imputación por el método de la media:
# Es un procedimiento sencillo, consiste en sustituir el valor ausente por la media de los valores válidos, presenta dos variantes:
# - Imputación por media no condicional: Consiste en estimar la media de los valores observados.
# - Imputación por media condicional: Consiste en agrupar los valores observados y no observados en clases e imputar los valores faltantes por la media de los valores observados en la misma clase.
# 
# #### - Imputación con variables ficticias
# Consiste en crear una variable indicador para identificar las observaciones con datos faltantes.
# #### - Imputación por regresión
# Consiste en estimar los valores ausentes en base a su relación con otras variables mediante análisis de regresión.Se incluyen en dicho grupo aquellos procedimientos de imputación que asignan a los campos a imputar valores en función del modelo: 
# 

# $$y_{vi}=α+β_{1} x_{1}+β_{2} x_{2}+...+β_{k} x_{k}
# +ε$$

# donde: $$y_{vi}$$ es la variable dependiente a imputar y las variables $${x_{j}|j≡1...n}$$ son las regresoras que pueden ser tanto cualitativas como cuantitativas, generalmente variables
# altamente correladas con la dependiente.

# ### Imputación múltiple
# + Es la imputación más recomendada.
# + Para la estimación de valores perdidos se realiza un modelo de regresión múltiple para predecir valores faltantes en función de múltiples variables.
# + Utiliza muestreos aleatorios de la distribución condicional de la variable de destino dadas otras variables.
# + La información que se añade para remplazar predecir los valores perdidos puede contener cualquier variable que sea potencialmente predictiva, incluyendo variables medidas a futuro.
# + Requiere especificarse en el modelo de regresión para observar cómo cambia la estimación.

# ## ¿Qué es la regresión lineal?

# La regresión lineal o ajuste lineal es un modelo matemático que se usa para hallar la aproximacion de la relación de dependencia de la variable **Y** con la variable **X**
# 
# 
# Es usada en muchos ámbitos para predecir un comportamiento que tenga que ver con 2 variables, en caso no se pueda aplicar se dice que no hay correlaci{on entre las variables estudiadas.

# **Regresión Lineal**: La regresión lineal se usa para encontrar una __`relación lineal entre el objetivo y uno o más predictores`__.

# ![que-es-la-regresion-lineal-y-para-que-sirve](https://user-images.githubusercontent.com/25250496/204172072-0fabbfdf-1c4c-4f9b-8f42-505d98b18b71.png)

# ## Escala de Likerd

# ### ¿Qué es la escala Likert?

# La escala de Likert es una escala de evaluación, lo que significa que el cuestionario de la escala de Likert consiste en respuestas cerradas y prellenadas, provistas de opciones numéricas o verbales. La escala de Likert es también una **escala multi elemento**, ya que consiste en una serie de afirmaciones (elementos) que expresan los tipos de actitudes que se quieren investigar.
# 
# 

# En un cuestionario Likert se utiliza una escala de 5 o 7 puntos, a veces llamada escala de satisfacción, ya que se pide al encuestado que exprese su grado de acuerdo/desacuerdo con cada declaración seleccionada. En la versión original usada por Likert las opciones de respuesta eran:

# **1** Totalmente de acuerdo
# 
# **2** De acuerdo
# 
# **3** Ni de acuerdo ni en desacuerdo
# 
# **4** En desacuerdo
# 
# **5** Totalmente en desacuerdo

# ### Ventajas de la escala de Likert
# 

# La **principal ventaja** de la escala de evaluación de Likert es que las respuestas pueden discernirse claramente y colocarse en la escala de valores para analizarlas rápidamente (las respuestas del encuestado ya corresponden a valores numéricos que pueden cuantificarse automáticamente).

# Otras ventajas son:
# 
# - El uso de la escala de Likert está muy extendido
# 
# - Codificación inmediata de las respuestas
# 
# - Suma automática de resultados
# 
# - Simplicidad de aplicación
# 
# - Compatible con las técnicas de cuantificación de datos
""")
st.write("""# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# <center> <h1>Propuesta</h1> </center> 

# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# ## 1.- Dataset

# <div class="alert alert-info">
# 
#     
#    Para poder tener el  <strong>Datset</strong> hay que recolectar información con una encuenta elaborada por nosotros.
#     
# </div>

 #### Encuesta ejemplo:

 La encuesta la realizamos en Google-Form donde se solicitara escoger un Dibujo Animado
 - Donde si escoge 1 es el que menos le gusta hasta 5 que es el que mas le gusta (escala de liker)

 #### Formulario de Google (Preguntas)

 En esta encuesta nos importa tu opinion sobre los lugares que te gustaria o no visitar, danos tu opinion segun esta escala: 1 (No, nunca), 2 (No me gustaria), 3 (Lo pensaria), 4 (Me gustaria), 5 (Si, me encantaria)

 ¿Te gustaria viajar al lugar donde se realizo la fotografia?

 #### Formulario de Google (Imagenes)

# ![Formulario](https://i.imgur.com/FKNCdqh.jpg)

# ![Formulario2](https://i.imgur.com/QqQYu2a.jpg)

#### Formulario de Google (Preprocesamiento)

# ![Formulario2](https://i.imgur.com/C1DJVz7.jpg)
""")
# In[2]:


#Importamos librerias para Ciencia de Datos y Machine Learning
import numpy as np
import pandas as pd

# nos ayuda a realizar graficas de calor
import seaborn as sns

# otra manera de grafic
import matplotlib.pyplot as plt


# In[3]:


#archivo CSV separado por comas
data = pd.read_csv('Formulario PIF.csv')

st.write("""## Datos obtenidos:""")
data


# In[5]:

st.write("""## Filas y columnas:""")
data.shape


# In[6]:


data.isnull().sum()


# In[7]:

st.write("""## Tipos de respuestas:""")
data.dtypes


st.write("""## Inputacion:""")

# In[8]:


data.describe()


# In[9]:


data.describe()
data["Desierto la Tatacoa"] = data["Desierto la Tatacoa"].replace(np.nan, 3)
data["Chichen Itza"] = data["Chichen Itza"].replace(np.nan, 4)
data["Laguna Azul"] = data["Laguna Azul"].replace(np.nan, 4)
data[" Angkor Wat"] = data[" Angkor Wat"].replace(np.nan, 4)
data_1=data
data_1


# In[10]:


data_1.isnull().sum()


## 2.- Correlación de Pearson  (Similitud)

# In[13]:

n = data_1[data_1.columns[1:]].to_numpy()
m = data_1[data_1.columns[0]].to_numpy()
print(n)
print(m)


st.write("""## 3.- Correlación en Pandas""")

# In[14]:


n.T


# In[15]:

try:
  respuestas = pd.DataFrame(n.T, columns = m)
  m_corr_pandas = respuestas.corr()
except ValueError:
  m_corr_pandas
  


# In[17]:


st.write("""![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)""")

## 4.- Matrix de Correlación

# In[20]:


matriz_pandas_4 = np.round(m_corr_pandas, 
                       decimals = 4)  
  


# In[21]:



# In[23]:


matriz_pandas_4.min().idxmin()


# In[24]:


matriz_pandas_4.sharonzalet.idxmin()


# In[25]:


maximo = matriz_pandas_4.unstack()
print (maximo.sort_values(ascending = False)[range(len(n),((len(n)+4)))])


st.write(""" ## Gráfica de Calor """)

# In[29]:


import seaborn as sns


# In[32]:


sns.heatmap(matriz_pandas_4,
            cmap= "viridis")


# <div class="alert alert-info">
# 
#     
#    **HALLAR**: a partir de la matriz de correlación en  <strong>Pandas</strong> .
#     
#    **Instalar** : `matplotlib` `seaborn`
#     
# </div>

st.write(""" ## 5.- RESULTADOS """)

st.write(""" Los resultados de similitud obtenidos en **Lugares a viajar** según la tabla de **Correlación** con los siguientes encuestados:

  1. larenaslo  y  raissasiza	obtienen el **PRIMER** indice mas bajo de similitud 
""")

# <div class="alert alert-info">
# 
#     
#    **HALLAR**: a partir de la matriz de correlación en  <strong>Pandas</strong>. A simple vista se puede observar los resultados, pero para una matriz mas grande se debe programar una `función` o `método` para que **localice los dos usuarios con mas alto valos de correlación**.
#     
# </div>

st.write(""" ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png) """)

st.write(""" <center> <h1>Validación de Resultados</h1> </center> """)

# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# ## Validación - Matrix de Correlación

# 
# <div class="alert alert-info">
# 
#    Se debe llenar la tabla de __VALIDACIÓN de la Matriz de Correlación__ con los valores de `Similitud` obtenidos
#     
#     
#    En `NUMPY` a partir de las matrices `n` y `m` con funciones.
#     
# </div>

# Se realiza la validación de los resultados obtenidos con la   `Matriz de Correlación de Pearson` en `Numpy` 
#  

# ### PROPUESTA

# In[33]:


n_2 = data_1[data_1.columns[1:]].to_numpy()
m_2 = data_1[data_1.columns[0]].to_numpy()
print(n_2)
print(m_2)


# In[34]:

rpta_pearson = []
def corr_pearson(a,b):
  ma = a.mean()
  mb = b.mean()
  numerador = np.sum((a -ma)*(b-mb))
  denominador = np.sqrt(np.sum((a-ma)** 2)*np.sum((b-mb)**2))
  return numerador / denominador
    
for i in range(len(m_2)):
  for j in range(len(m_2)):
      data_3 = data_1.loc[[i,j],:]
      nuevo = data_3[data_3.columns[1:]].to_numpy()
      rpta_pearson.append(corr_pearson(nuevo[0],nuevo[1]))
   
rpta_final_pearson = np.array(rpta_pearson).reshape(len(m_2),len(m_2))
resultado = pd.DataFrame(rpta_final_pearson,m_2,m_2)
resultado
    


maximo_2 = resultado.unstack()
print (maximo_2.sort_values(ascending = False)[range(len(n_2),((len(n_2)+4)))])

st.write("""## Resultados """)

st.write(""" Los resultados de similitud obtenidos en **Lugares a viajar** según la tabla de **Correlación** con los siguientes encuestados:

  1. larenaslo y raissasiza obtienen el **PRIMER** indice mas bajo de similitud 
""")
# In[36]:


sns.heatmap(resultado,
            cmap= "hot")


# In[41]:


plt.scatter(resultado["raissasiza"],resultado["larenaslo"])
plt.xlabel("raissasiza")
plt.ylabel("larenaslo")
plt.title("Relacion entre raissasiza y larenaslo")
plt.show


# In[42]:


plt.scatter(resultado["sharonzalet"],resultado["makarenavaldeiglesiassota"])
plt.xlabel("sharonzalet")
plt.ylabel("makarenavaldeiglesiassota")
plt.title("Relacion entre sharonzalet y makarenavaldeiglesiassota")
plt.show


st.write(""" ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png) """)

st.write(""" <center> <h1>Conclusiones</h1> </center> """)

#  <div class="alert alert-info">
#     
#    - ¿Se valido o no los resultados?
#    - Los resultados Validados son:
#    - ¿Es efectivo el metodo de correlación de pearson?
#    - Correlación de Pearson y Regresión Lineal, ¿cual es su relación?
#     
#  </div>

st.write(""" ## ¿Se valido o no los resultados?
# 
# Sí se valido los resultados

# ## Los resultados validados son: 
# Al hallar con la Correlación de Pearson, el primer correo **gzapanaca** y el segundo correo **makarenavaldeiglesiassota** nos dio una respuesta: y hallandolo manualmente llegamos al mismo resultado con la ayuda de df_corr.
# 

# ## ¿Es efectivo el método de correlación de Pearson?
# **SI** es efectivo usar el método de correlación de Pearson porque son valores atípicos.

# ## Correlación de Pearson y Regresión Lineal, ¿cuál es su relación?
# 

# La  correlación de Pearson se usa para establecer la relación entre 2 variables, con la regresión lineal se obtiene la ecuación de dependencia entre varias variables.

# ![linea 1](https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png)

# ## Referencia

# - __Profesor de Matematicas__: `John Gabriel Muñoz Cruz`
# https://www.linkedin.com/in/jgmc
# - __TuDashboard__ (28 de Junio de 2021):
# https://tudashboard.com/grafica-de-dispersion/
# - __Youtube__: `Pro ciencia` 
# https://www.youtube.com/watch?v=UwXWy5OVvU8
# - __Investigacion__ `D´eborah Otero `
# http://eio.usc.es/pub/mte/descargas/ProyectosFinMaster/Proyecto_616.pdf
# - __Microdepuración e imputación de datos__
# https://www.ugr.es/~diploeio/documentos/tema6.pdf
# - __Correlación Linel de Pearson__python
# https://www.cienciadedatos.net/documentos/pystats05-correlacion-lineal-python.html
# - __Analisis de correlación lineal__
# http://halweb.uc3m.es/esp/Personal/personas/jmmarin/esp/GuiaSPSS/17corlin.pdf
# - __COeficiente de correlación__
# https://hmong.es/wiki/Pearson_product-moment_correlation_coefficient

# ![linea 1](https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png)
""")
</h1>
# %%
