import  streamlit as st  
import  pandas  as  pd
import  numpy as np  

lista  = [ 'nchambillac@unsa.edu.pe' ]
if st.experimental_user.email in lista: 
    st.escribir ( 'Bienvenido' , st.experimental_user.email )
    

menu = st.sidebar.markdown("<h2 style='text-align: ; color: black;'>Menu</h2>", unsafe_allow_html=True)
inicio = st.sidebar.button('Inicio')
objetivos = st.sidebar.button('Objetivos')
base_teorica = st.sidebar.button('Base Teorica')
propuesta = st.sidebar.button('Propuesta')
correlacion_de_pearson = st.sidebar.button('Correlacion de Pearson')
nuestra_correlacion = st.sidebar.button('Nuestra Correlacion')
mapa_de_calor = st.sidebar.button('Mapa de Calor')
validacion_de_resultados = st.sidebar.button('Validacion de Resultados')
conclusiones = st.sidebar.button('Conclusiones')


if  inicio :
    
    st.markdown ( "<h2 style='text-align: center; color: black;'>Universidad Nacional de San Agustín de Arequipa </h2>" , unsafe_allow_html = True )
    st.markdown ( "<h2 style='text-align: center; color: black;'>Escuela Profesional de Ingeniería de Telecomunicaciones </h2>" , unsafe_allow_html = True )

    st.imagen ( 'https://www.unsa.edu.pe/wp-content/uploads/sites/3/2018/05/Logo-UNSA.png' )

    st.markdown ( "<h2 style='text-align: center; color: black;'>Ingeniero Renzo Bolivar - Docente DAIE</h2>" , unsafe_allow_html = True )

    st.markdown ( "<h2 style='text-align: center; color: black;'>Curso : Computación 1 </h2>" , unsafe_allow_html = True )

    st.imagen ( 'https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png' )

    st.markdown ( "<h2 style='text-align: center; color: black;'>GRUPO A </h2>" , unsafe_allow_html = True )

    st.markdown ( "<h2 style='text-align: center; color: black;'>Alumnos:</h2>" , unsafe_allow_html = True )

    st.markdown ( "<h2 style='text-align: center; color: black;'>Chambilla Cayo,Nohelia Kathia</h2>" , unsafe_allow_html = True )   
     

    st.imagen ( 'https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png' )

    st.markdown ( "<h2 style='text-align: center; color: black;'>INVESTIGACIÓN FORMATIVA</h2>" , unsafe_allow_html = True )
  
    st.markdown ( "<h2 style='text-align: center; color: black;'>PYTHON - Inteligencia Artificial</h2>" , unsafe_allow_html = True )
    
    
    
    st.imagen ( 'https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png' )
   
if  objetivos :
    
    st.imagen ( 'https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png' )
    st.markdown ( "<h2 style='text-align: color: black;'>Los objetivos de la investigación formativa son:</h2>" , unsafe_allow_html = True )

    st.markdown ( '1. Competencia Comunicativa Presentación de sus resultados con lenguaje de programación Python utilizando los archivos Jupyter Notebook.' )


    st.markdown ( '2. Competencia Aprendizaje: con las aptitudes en Descomposición (desarticular el problema en pequeñas series de soluciones), Reconocimiento de Patrones (encontrar simulitud al momento de resolver problemas), Abstracción (omitir información relevante), Algoritmos (pasos para resolución de un problema)' )

    st.markdown ( '3. Competencia de Trabajo en Equipo: exige habilidades individuales y grupales orientadas a la cooperación, planificación, coordinación, prevista de tareas, cumplimiento de tareas y solución de conflictos en pro de un trabajo colectivo, utilizando los archivos Jupyter Notebook los cuales se sincronizan en el servidor Gitlab con comandos Git.' )

    st.imagen ( 'https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png' )

    st.markdown ( "<h2 style='text-align: center; color: black;'>Aplicación en IA </h2>" , unsafe_allow_html = True )

    st.imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    
si  base_teorica :
    
    st.imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    
    st.markdown ( "<h2 style='text-align: center; color: black;'>Base Teórica</h2>" , unsafe_allow_html = True )

    st.markdown ( "<h2 style='text-align: color: black;'>Análisis de Correlación</h2>" , unsafe_allow_html = True )

    st.markdown ( 'El análisis de conexiones es el primer paso para construir modelos explicativos y predictivos más complejos.' )

 
    st.markdown ( 'A menudo nos interesa observar y medir la relación entre 2 variables numéricas mediante el análisis de transmisión.' )

    st.markdown ( 'Se trata de una de las técnicas más habituales en análisis de datos y el primer paso necesario antes de construir cualquier modelo explicativo o predictivo más complejo.' )

    st.markdown ( 'Para poder tener el Datset hay que recolectar información a travez de encuentas.' )

    st.markdown ( "<h2 style='color: black;'>¿Qué es la conexión?</h2>" , unsafe_allow_html = True )

    st.markdown ( 'La coincidencia es un tipo de asociación entre dos variables numéricas, específicamente evalúa la tendencia (creciente o decreciente) en los datos.' )

    st.markdown ( 'Dos variables están asociadas cuando una variable nos da información acerca de la otra. Por el contrario, cuando no existe asociación, el aumento o disminución de una variable no nos dice nada sobre el comportamiento de la otra variable.' )
 
    st.markdown ( 'Dos variables se correlacionan cuando muestran una tendencia creciente o decreciente.' )

    st.markdown ( "<h2 style='text-align: color: black;'>¿Cómo se mide la conexión?</h2>" , unsafe_allow_html = True )

    st.markdown ( 'Tenemos el coeficiente de correlación lineal de Pearson que se sirve para cuantificar tendencias lineales, y el coeficiente de correlación de Spearman que se utiliza para tendencias de aumento o disminución, no obstante lineales pero sí monótonas.' )

    st.markdown ( "<h2 style='text-align: color: black;'>Correlación de Pearson</h2>" , unsafe_allow_html = True )


    st.markdown ( 'El coeficiente de correlación lineal de Pearson mide una tendencia lineal entre dos variables numéricas.' )

    st.markdown ( 'Es el método de conexión más utilizado, pero asume que:.' )
    st.markdown ( '- La tendencia debe ser de tipo lineal.' )
    st.markdown ( 'No existen valores atípicos (outliers' )
    st.markdown ( '- Las variables deben ser numéricas.' )
    st.markdown ( '- Tenemos suficientes datos (algunos autores recomiendan tener más de 30 puntos u observaciones).' )

    st.markdown ( 'Los dos primeros supuestos se pueden evaluar simplemente con un diagrama de dispersión, mientras que para los últimos basta con mirar los datos y evaluar el diseño que tenemos.' )

    st.markdown ( "<h2 style='text-align: color: black;'>Cómo se interpreta la conexión</h2>" , unsafe_allow_html = True )

    st.markdown ( 'El signo nos indica la dirección de la relación, como hemos visto en el diagrama de dispersión.' )
    st.markdown ( '- un valor positivo indica una relación directa o positiva.' )
    st.markdown ( '- un valor negativo indica relación indirecta, inversa o negativa.' )
    st.markdown ( ' - un valor nulo indica que no existe una tendencia entre ambas variables (puede ocurrir que no exista relación o que la relación sea más compleja que una tendencia, por ejemplo, una relación en forma de U).' )


    st.markdown ( 'La magnitud nos indica la fuerza de la relación, y toma valores entre $-1$ a $1$. Cuanto más cercano sea el valor a los extremos del intervalo ($1$ o $-1$) más fuerte será la tendencia de las variables, o será menor la dispersión que exista en los puntos alrededor de dicha tendencia .')
    st.markdown ( '- si la conexión vale $1$ o $-1$ diremos que la conexión es “perfecta”.' )
    st.markdown ( '- si la conexión vale $0$ diremos que las variables no están correlacionadas.' )

    st.imagen ( "https://user-images.githubusercontent.com/25250496/204172549-2ccf3be3-a2b3-4b49-9cd4-adb66e28621d.png" )

    st.markdown ( "<h2 style='text-align: center; color: black;'>Fórmula Coeficiente de Correlación de Pearson</h2>" , unsafe_allow_html = True )

    st.imagen ( 'https://www.questionpro.com/blog/wp-content/uploads/2019/07/analisis-de-correlacion-2.jpg' )


    st.markdown ( 'Distancia Euclidiana: La distancia euclidiana es la generalización del teorema de Pitágoras.' )

    st.imagen ( 'http://3.bp.blogspot.com/-YU7iwu9rMIk/TuZLXuWM_8I/AAAAAAAAGNU/DqsQeCkheMk/s1600/Pit%25C3%25A1goras.png' )

    st.markdown ( 'Regresión Lineal: La regresión lineal se usa para encontrar una relación lineal entre el objetivo y uno o más predictores.' )

    st.imagen ( 'https://user-images.githubusercontent.com/25250496/204172072-0fabbfdf-1c4c-4f9b-8f42-505d98b18b71.png' )

    st.markdown ( "<h2 style='text-align: color: black;'>Marco teórico</h2>" , unsafe_allow_html = True )

    st.markdown ( "<h2 style='text-align: color: black;'>¿Que es machine learning?</h2>" , unsafe_allow_html = True )

    st.markdown ( 'Machine learning: Es una rama de la inteligencia artificial que es autónoma ya que puede realizar predicciones a partir de procesamiento de datos. Esta tecnología está presente en un sinfín de aplicaciones como las recomendaciones de Netflix o Spotify, las respuestas inteligentes de Gmail o el habla de Siri y Alexa. El ‘machine learning’ es un maestro del reconocimiento de patrones, y es capaz de convertir una muestra de datos en un programa informático capaz de extraer inferencias de nuevos conjuntos de datos para los que no ha sido entrenado previamente Aunque ahora esté de moda, gracias a su capacidad para derrotar a jugadores del Go o resolver cubos de Rubik, su origen se remonta al siglo pasado. “La estadística es sin duda la base fundamental del aprendizaje automático, que básicamente consiste en una serie de algoritmos capaces de analizar grandes cantidades de datos para deducir cuál es el resultado más óptimo para un determinado problema”, añade Espinoza.' )
    

    st.imagen ( 'https://www.atriainnovation.com/wp-content/uploads/2021/02/portada.jpg' )

    st.markdown ( "<h2 style='text-align: color: black;'>LIBRERIAS PARA CIENCIA DE DATOS:</h2>" , unsafe_allow_html = True )


    

    st.markdown ( 'Conjuntos de archivos de código que han sido creados para desarrollar software de manera sencilla. Gracias a ellas, los desarrolladores pueden evitar la duplicidad de código y minimizar errores con mayor agilidad .Estas librerías son altamente prácticas a la hora de implementar flujos de Machine Learning.' )

    

    st.markdown ( "<h2 style='text-align: color: black;'>¿Qué es NUMPY?</h2>" , unsafe_allow_html = True )
            
    
    st.markdown ( 'Es una librería de Python especializada en el cálculo numérico y el análisis de datos, especialmente para un gran volumen de datos.Incorpora arrays lo que permite representar colecciones de datos de un mismo tipo en varias dimensiones, y funciones muy eficientes para su manipulación.Además cuenta con múltiples herramientas para manejar matrices de una forma muy eficiente.')

    st.imagen ('https://aprendeconalf.es/docencia/python/manual/img/arrays.png')
    
    st.markdown ('"<h2 style='text-align: color: black;'>¿Qué es PANDAS?</h2>" , unsafe_allow_html = True ')                

    st.markdown ( 'Pandas es una librería de código abierto que ofrece unas estructuras muy poderosas y flexibles que facilitan la manipulación y tratamiento de datos.Las principales funciones de pandas son :cargar datos, modelar, analizar, manipular y prepararlos.' )

    st.imagen ('https://profile.es/wp-content/media/Estructuras-de-datos-en-Pandas-1.png')
    
    st.markdown ( "<h2 style='text-align: color: black;'>¿Qué es matplotlib?</h2>" , unsafe_allow_html = True  )
    st.markdown ('Es una biblioteca para la generación de gráficos en dos dimensiones, a partir de datos contenidos en listas en el lenguaje de programación Python. Permite crear y personalizar los tipos de gráficos más comunes, entre ellos:')
    st.markdown ( '- Diagramas de barras' )
    st.markdown ( '- Histogramas' )
    st.markdown ( '- Diagramas de sectores' )
    st.markdown ( '- Diagramas de areas' )

    st.imagen ( 'https://matplotlib.org/3.1.1/_static/logo2_compressed.svg' )

    st.imagen ('https://miro.medium.com/max/1400/1*JTEqCz-VU16nkkUwzyWp_w.png')
    
    st . markdown ( "<h2 style='text-align: color: black;'>Seaborn:</h2>" , unsafe_allow_html = True )

    st . markdown ( 'Es una librería para Python que permite generar fácilmente gráficos. Seaborn está basada en matplotlib y proporciona una interfaz de alto nivel . Tiene como objetivo convertir la visualización en una parte central de la exploración y comprensión de los datos, generando atractivas gráficas con sencillas funciones que ofrecen una interfaz semejante, facilitando el paso de unas funciones a otras.' )
    

    st . imagen ( 'https://livecodestream.dev/post/how-to-build-beautiful-plots-with-python-and-seaborn/featured.jpg' )

    st . markdown ( "<h2 style='text-align: color: black;'>¿Qué es la escala Likert?</h2>" , unsafe_allow_html = True )

    st . markdown ( 'Se utiliza para medir qué tan de acuerdo están los encuestados con una variedad de afirmaciones.Esta es ideal para medir reacciones, actitudes y comportamientos de una persona. A diferencia de una simple pregunta de “sí” / “no”, la escala de Likert permite a los encuestados calificar sus respuestas.' )

    st . markdown ( 'La escala de Likert es uno de los tipos de escalas de medición utilizados principalmente en la investigación de mercados para la comprensión de las opiniones y actitudes de un consumidor hacia una marca, producto o mercado meta. Nos sirve principalmente para realizar mediciones y conocer sobre el grado de conformidad de una persona o encuestado hacia determinada oración afirmativa o negativa.' )
    st . markdown ( 'Las respuestas pueden ser ofrecidas en diferentes niveles de medición, permitiendo escalas de 5, 7 y 9 elementos configurados previamente. Siempre se debe tener un elemento neutral para aquellos usuarios que ni de acuerdo ni en desacuerdo.' )

    st . markdown (  "<h2 style='text-align: color: black;'>CSV (Valores Separados por Comas):</h2>" , unsafe_allow_html = True  )


    st . markdown ( 'Es el formato más común de importación y exportación de hojas de cálculo y bases de datos. Es cualquier archivo de texto en el cual los caracteres están separados por comas, haciendo una especie de tabla en filas y columnas. Las columnas quedan definidas por cada punto y coma (;), mientras que cada fila se define mediante una línea adicional en el texto. De esta manera, se pueden crear archivos CSV con gran facilidad.' )

    st . imagen ( 'https://acf.geeknetic.es/imgri/imagenes/tutoriales/definiciones/2020/6/Archivo-CSV-qrdy.jpg?f=webp' )

    
    
    st . markdown ( "<h2 style='text-align: color: black;'>DATOS FALTANTES</h2>" , unsafe_allow_html = True )

   
    st . markdown ('Son aquellos que no constan debido a cualquier acontecimiento, como por ejemplo errores en la transcripción de los datos o la ausencia de disposición a responder a ciertas cuestiones de una encuesta. Los datos pueden faltar de manera aleatoria o no aleatoria.')

    st . markdown ("<h2 style='text-align: color: black;'>METODOS DE IMPUTACION</h2>" , unsafe_allow_html = True )

    st . markdown ('Los métodos de imputación consisten en estimar los valores ausentes en base a los valores válidos de otras variables y/o casos de la muestra. La estimación se puede hacer a partir de la información del conjunto completo de variables o bien de algunas variables especialmente seleccionadas. Usualmente los métodos de imputación se utilizan con variables métricas (de intervalo o de razón), y deben aplicarse con gran precaución porque pueden introducir relaciones inexistentes en los datos realas.')

    

    st . markdown ('Sustitución por la Media. Consiste en sustituir el valor ausente por la Media de los valores válidos. Este procedimiento plantea inconvenientes como:')
    st . markdown ('-Dificulta la estimación de la Variáncia.')

    st . markdown ('-Distorsiona la verdadera distribución de la variable,')
    st . markdown ('Distorsiona la correlación entre variables dado que añade valores constantes.')

    st . markdown ('Sustitución por constante. Consiste en sustituir los valores ausentes por constantes cuyo valor viene determinado por razones teóricas o relacionadas con la investigación previa. Presenta los mismos inconvenientes que la sustitución por la Media, y solo debe ser utilizado si hay razones para suponer que es más adecuado que el método de la media.')

    st . markdown ("<h2 style='text-align: color: black;'>¿Qué es un framework?</h2>" , unsafe_allow_html = True )

    st . markdown ('Es una especie de plantilla, un esquema conceptual, que simplifica la elaboración de una tarea, ya que solo es necesario complementarlo de acuerdo a lo que se quiere realizar.')

    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )

si  propuesta :
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: center; color: black;'>Propuesta</h2>" , unsafe_allow_html = True )

    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )

    st . markdown ( "<h2 style='color: black;'>1.- Conjunto de datos</h2>" , unsafe_allow_html = True )

    st . markdown ( "<h2 style='color: black;'>Formulario de Google </h2>" , unsafe_allow_html = True )

    st . markdown ( "<h2 style='color: black;'>Preguntas </h2>" , unsafe_allow_html = True )

    st . markdown ( 'Del uno al cinco segun la escala dada se califico las siguientes canciones:' )

 
    st . markdown ( 'Formulario de Google' )


    st . imagen ( 'https://scontent.faqp2-1.fna.fbcdn.net/v/t39.30808-6/320803430_2534042946737581_4638513726078409310_n.jpg?_nc_cat=106&ccb=1-7&_nc_sid=730e14&_nc_eui2=AeHuPW9i0i057wIeGX31Tjg6zub5xqJ23AjO5vnGonbcCDVYrVZM6f8Qfwdo6WF066Bx71BU41bEuQsTDK2aPU2b&_nc_ohc=VvDQm8k7MtgAX_Fuetx&_nc_ht=scontent.faqp2-1.fna&oh=00_AfBvcWkI92TTgkUoK3ZjeHc7qUac0isg9VDfycxDPJBjyw&oe=63A9691E' )
    st . imagen ( 'https://scontent.faqp2-2.fna.fbcdn.net/v/t39.30808-6/321392210_713012620341312_1618897939528560238_n.jpg?_nc_cat=101&ccb=1-7&_nc_sid=730e14&_nc_eui2=AeFSXYlXfZko7t_S_2gGlMKZQtNjPu8iV9FC02M-7yJX0YiSea2eJ7bWXn10WIUgoXCFPintattSSYW9-rxB4dLx&_nc_ohc=lcnUCmyPtqEAX-poU8h&_nc_ht=scontent.faqp2-2.fna&oh=00_AfC2ELvwmm1Z_9SDuZSrbfaGjdN04HaItgZxUOeVT9v0hg&oe=63AA8028' )
   

    st . markdown ( "<h2 style='color: black;'>PFormulario de Google (Preprocesamiento) </h2>" , unsafe_allow_html = True )

    st . markdown ( 'Procedemos a crear nuestro exel para luego ser convertido en un archivo csv con el fin de utilizarlo junto al comando panda y numpy.' )

    st . imagen ( 'https://scontent.faqp2-2.fna.fbcdn.net/v/t39.30808-6/320883798_857727338791655_4101368769266917084_n.jpg?_nc_cat=104&ccb=1-7&_nc_sid=730e14&_nc_eui2=AeETD5uRsjg_dxuDwF_JnbWyfOJE7kY4m7584kTuRjibvlm73KkXs874kOqskQVA5zUooM-RQSBgw-K9YkE1JiXR&_nc_ohc=wVRttRX5jLQAX_uWnhH&_nc_ht=scontent.faqp2-2.fna&oh=00_AfDG90eMleVW9Z0W8u6abWZjOUgu70j5fkT7UZ0AsOkXQA&oe=63A9649D' )
    
    st . markdown ( "<h2 style='text-align: center; color: black;'>Archivo CSV separado por comas</h2>" , unsafe_allow_html = True )
    pandas  =  pd . read_csv ( 'ENCUESTA.csv' )
    st . marco de datos ( pandas )
    st . markdown ( 'Cantidad de Filas y Columnas' )
    
    st . markdown ( '58 Columnas y 37 Filas' )
    
    
si  correlacion_de_pearson :
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: center; color: black;'>Correlacion de Pearson y Sustitución de valores NAN</h2>" , unsafe_allow_html = True )
    pandas  =  pd . read_csv ( 'ENCUESTA.csv' )
    st . markdown ( 'Se visualizan los valores NAN que serán imputados en el dataframe' )
    st . marco de datos ( pandas )
    st . markdown ( 'Se muestra donde se encuentran los valores NAN (float64)' )
    pandas _ tipos de d
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: center; color: black;'>Tabla limpia de valores NAN</h2>" , unsafe_allow_html = True )
    st . markdown ( 'Los valores que reemplazan los NAN son la media' )
    data['CREO EN TI']=data['CREO EN TI'].replace(np.nan,4)
    data['COLGANDO EN TUS MANOS']=data['COLGANDO EN TUS MANOS'].replace(np.nan,4)
    data['MAMIII']=data['MAMIII'].replace(np.nan,3)
    data['HEY MOR']=data['HEY MOR'].replace(np.nan,3)
    data['DANZA KUDURO']=data['DANZA KUDURO'].replace(np.nan,4)
    data['DESPECHA']=data['DESPECHA'].replace(np.nan,3)
    data['QUE MAS PUES ']=data['QUE MAS PUES '].replace(np.nan,3)
    data['UN VERANO SIN TI']=data['UN VERANO SIN TI'].replace(np.nan,3)
    data['NINACHAY']=data['NINACHAY'].replace(np.nan,5)
    data['POR USTEDES']=data['POR USTEDES'].replace(np.nan,4)
    data['MATADOR']=data['MATADOR'].replace(np.nan,3)
    data['MONTONERO AREQUIPENO']=data['MONTONERO AREQUIPENO'].replace(np.nan,4)
    data['MALA TU']=data['MALA TU'].replace(np.nan,3)
    data['CON LA BRISA ']=data['CON LA BRISA '].replace(np.nan,3)
    data['MY LIFE ']=data['MY LIFE '].replace(np.nan,4)
    data['SIGN OF THE TIMES ']=data['SIGN OF THE TIMES '].replace(np.nan,4)
    data['BLOODY MARY']=data['BLOODY MARY'].replace(np.nan,4)
    data['PROPUESTA INDECENTE']=data['PROPUESTA INDECENTE'].replace(np.nan,4)
    st . marco de datos ( pandas )
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: center; color: black;'>Tabla de Correlacion de Pandas</h2>" , unsafe_allow_html = True )
    n = data[data1.columns[1:]].to_numpy()
    m = data[data1.columns[0]].to_numpy()
    df = pd.DataFrame(n.T, columns = m)

    m_corr_pandas = df.corr()
    m_corr_pandas
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    
    
si  nueva_correlacion :
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    pandas  =  pd . read_csv ( 'ENCUESTA.csv' )
    st . markdown ( "<h2 style='text-align: center; color: black;'>Nuestro Algoritmo y su Respectivo Grafico</h2>" , unsafe_allow_html = True )
    st . markdown ( 'Para esta parte hemos tomado el dataframe limpio de valores NAN, para poder realizar nuestro algoritmo de correlacion, empezamos eliminando la columna de correo electronico para solo tener un dataframe de valores numericos.' )
    data['CREO EN TI']=data['CREO EN TI'].replace(np.nan,4)
    data['COLGANDO EN TUS MANOS']=data['COLGANDO EN TUS MANOS'].replace(np.nan,4)
    data['MAMIII']=data['MAMIII'].replace(np.nan,3)
    data['HEY MOR']=data['HEY MOR'].replace(np.nan,3)
    data['DANZA KUDURO']=data['DANZA KUDURO'].replace(np.nan,4)
    data['DESPECHA']=data['DESPECHA'].replace(np.nan,3)
    data['QUE MAS PUES ']=data['QUE MAS PUES '].replace(np.nan,3)
    data['UN VERANO SIN TI']=data['UN VERANO SIN TI'].replace(np.nan,3)
    data['NINACHAY']=data['NINACHAY'].replace(np.nan,5)
    data['POR USTEDES']=data['POR USTEDES'].replace(np.nan,4)
    data['MATADOR']=data['MATADOR'].replace(np.nan,3)
    data['MONTONERO AREQUIPENO']=data['MONTONERO AREQUIPENO'].replace(np.nan,4)
    data['MALA TU']=data['MALA TU'].replace(np.nan,3)
    data['CON LA BRISA ']=data['CON LA BRISA '].replace(np.nan,3)
    data['MY LIFE ']=data['MY LIFE '].replace(np.nan,4)
    data['SIGN OF THE TIMES ']=data['SIGN OF THE TIMES '].replace(np.nan,4)
    data['BLOODY MARY']=data['BLOODY MARY'].replace(np.nan,4)
    data['PROPUESTA INDECENTE']=data['PROPUESTA INDECENTE'].replace(np.nan,4)
    
    n = data[data1.columns[1:]].to_numpy()
    m = data[data1.columns[0]].to_numpy()
    df = pd.DataFrame(n.T, columns = m)

    m_corr_pandas = df.corr()
    m_corr_pandas
    
    con  st . eco ():       
        marco de datos1  =  pd . Marco de datos ( pandas )
        corr_Diferente = []

    def matriz_corr(x,y):
        mx = x.mean()
        my = y.mean()
        n = np.sum((x -mx)*(y-my))
        d = np.sqrt(np.sum((x-mx)** 2)*np.sum((y-my)**2))
        return n / d

    for i in range(len(m2)):
        for j in range(len(m2)):
            data2 = data1.loc[[i,j],:]
            nuevo = data2[data2.columns[1:]].to_numpy()
            corr_Diferente.append(matriz_corr(nuevo[0],nuevo[1]))
        
    correlacion_final = np.array(corr_Diferente).reshape(len(m2),len(m2))
    respuesta = pd.DataFrame(correlacion_final,m2,m2)
    respuesta
    
    st . dataframe ( dataframe_correlacion_pearson )
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: center; color: black;'>Mayor Valor de Correlacion</h2>" , unsafe_allow_html = True )
    data['CREO EN TI']=data['CREO EN TI'].replace(np.nan,4)
    data['COLGANDO EN TUS MANOS']=data['COLGANDO EN TUS MANOS'].replace(np.nan,4)
    data['MAMIII']=data['MAMIII'].replace(np.nan,3)
    data['HEY MOR']=data['HEY MOR'].replace(np.nan,3)
    data['DANZA KUDURO']=data['DANZA KUDURO'].replace(np.nan,4)
    data['DESPECHA']=data['DESPECHA'].replace(np.nan,3)
    data['QUE MAS PUES ']=data['QUE MAS PUES '].replace(np.nan,3)
    data['UN VERANO SIN TI']=data['UN VERANO SIN TI'].replace(np.nan,3)
    data['NINACHAY']=data['NINACHAY'].replace(np.nan,5)
    data['POR USTEDES']=data['POR USTEDES'].replace(np.nan,4)
    data['MATADOR']=data['MATADOR'].replace(np.nan,3)
    data['MONTONERO AREQUIPENO']=data['MONTONERO AREQUIPENO'].replace(np.nan,4)
    data['MALA TU']=data['MALA TU'].replace(np.nan,3)
    data['CON LA BRISA ']=data['CON LA BRISA '].replace(np.nan,3)
    data['MY LIFE ']=data['MY LIFE '].replace(np.nan,4)
    data['SIGN OF THE TIMES ']=data['SIGN OF THE TIMES '].replace(np.nan,4)
    data['BLOODY MARY']=data['BLOODY MARY'].replace(np.nan,4)
    data['PROPUESTA INDECENTE']=data['PROPUESTA INDECENTE'].replace(np.nan,4)

   
    n = data[data1.columns[1:]].to_numpy()
    m = data[data1.columns[0]].to_numpy()
    df = pd.DataFrame(n.T, columns = m)

    m_corr_pandas = df.corr()
    
    m_corr_p = np.round(m_corr_pandas, decimals = 4)
    
    con  st . eco ():   
    def tidy_corr_matrix(corr_pandas):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy.
    '''
        corr_pandas = corr_pandas.stack().reset_index()
        corr_pandas.columns = ['Datos1','Datos2','relacion']
        corr_pandas = corr_pandas.loc[corr_pandas['Datos1'] != corr_pandas['Datos2'], :]
        corr_pandas['may a men'] = np.abs(corr_pandas['relacion'])
        corr_pandas = corr_pandas.sort_values('may a men', ascending=False)
    
        return(corr_pandas)

    tidy_corr_matrix(m_corr_p).head(20)
    
    st . escribir ( ordenar )   
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    
si  mapa_de_calor :
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    pandas  =  pd . read_csv ( 'ENCUESTA.csv' )
    marco de datos1  =  pd . Marco de datos ( pandas )
    lista1  =  dataframe1 [ 'Dirección de correo electrónico' ]. tolista ()
    data['CREO EN TI']=data['CREO EN TI'].replace(np.nan,4)
    data['COLGANDO EN TUS MANOS']=data['COLGANDO EN TUS MANOS'].replace(np.nan,4)
    data['MAMIII']=data['MAMIII'].replace(np.nan,3)
    data['HEY MOR']=data['HEY MOR'].replace(np.nan,3)
    data['DANZA KUDURO']=data['DANZA KUDURO'].replace(np.nan,4)
    data['DESPECHA']=data['DESPECHA'].replace(np.nan,3)
    data['QUE MAS PUES ']=data['QUE MAS PUES '].replace(np.nan,3)
    data['UN VERANO SIN TI']=data['UN VERANO SIN TI'].replace(np.nan,3)
    data['NINACHAY']=data['NINACHAY'].replace(np.nan,5)
    data['POR USTEDES']=data['POR USTEDES'].replace(np.nan,4)
    data['MATADOR']=data['MATADOR'].replace(np.nan,3)
    data['MONTONERO AREQUIPENO']=data['MONTONERO AREQUIPENO'].replace(np.nan,4)
    data['MALA TU']=data['MALA TU'].replace(np.nan,3)
    data['CON LA BRISA ']=data['CON LA BRISA '].replace(np.nan,3)
    data['MY LIFE ']=data['MY LIFE '].replace(np.nan,4)
    data['SIGN OF THE TIMES ']=data['SIGN OF THE TIMES '].replace(np.nan,4)
    data['BLOODY MARY']=data['BLOODY MARY'].replace(np.nan,4)
    data['PROPUESTA INDECENTE']=data['PROPUESTA INDECENTE'].replace(np.nan,4)    
    n = data[data1.columns[1:]].to_numpy()
    m = data[data1.columns[0]].to_numpy()
    df = pd.DataFrame(n.T, columns = m)

    m_corr_pandas = df.corr()
    
    m_corr_p = np.round(m_corr_pandas, decimals = 4)  

    def tidy_corr_matrix(corr_pandas):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy.
    '''
        corr_pandas = corr_pandas.stack().reset_index()
        corr_pandas.columns = ['Datos1','Datos2','relacion']
        corr_pandas = corr_pandas.loc[corr_pandas['Datos1'] != corr_pandas['Datos2'], :]
        corr_pandas['may a men'] = np.abs(corr_pandas['relacion'])
        corr_pandas = corr_pandas.sort_values('may a men', ascending=False)
    
        return(corr_pandas)

    tidy_corr_matrix(m_corr_p).head(20) 

    st . markdown ( "<h2 style='text-align: center; color: black;'>Mapa de Calor</h2>" , unsafe_allow_html = True )
    st . markdown ( 'En esta sección se mostrarán los mapas de calor obtenidos mediante la libreria pandas y el obtenido desde nuestro algoritmo. Comparando ambas para poder confirmar su similitud.' )
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align:; color: black;'>Mapa de Calor de Pandas</h2>" , unsafe_allow_html = True )
    st . image ( 'https://scontent.faqp2-1.fna.fbcdn.net/v/t39.30808-6/321675318_717252202968243_2569053303383691197_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=730e14&_nc_eui2=AeEp9yH_CLdoWx0JxCQD85nTfs03RmKri0R-zTdGYquLROkMmyuHseJ2IaxFcvb8XV9OjFiARttfxMWdFHUJOC7h&_nc_ohc=mflamIikrc8AX9KMlSD&_nc_ht=scontent.faqp2-1.fna&oh=00_AfDH3PLT6SfPxwo4JIMp6Ofa55WxuXdvldtPk3AQuVmBbA&oe=63AA0BAC' )
    st . markdown ( "<h2 style='text-align:; color: black;'>Mapa de Calor personal </h2>" , unsafe_allow_html = True )
    st . image ( 'https://scontent.faqp2-1.fna.fbcdn.net/v/t39.30808-6/321099572_1346330802807205_1098137571726666278_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=730e14&_nc_eui2=AeGThkcpoboG--4wRn__tlNAY1I3LkboesRjUjcuRuh6xFyawAFnMEVR2Cqdz1hV1xzWjjIPnQP6D4j2TGfoNnWt&_nc_ohc=CbDIkOSUboMAX8hPrLp&_nc_ht=scontent.faqp2-1.fna&oh=00_AfC4PQALm1uRiYpakveqfqO8bTB6-6j0dfK0E8qPY2wOvA&oe=63AA2F26' )
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )


si  validacion_de_resultados :
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: center; color: black;'>Validación de Resultados</h2>" , unsafe_allow_html = True )
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: ; color: black;'>Validacion-Matrix de Correlacion</h2>" , unsafe_allow_html = True )
    st . markdown ( 'Se debe llenar la tabla de VALIDACIÓN de la Matriz de Correlación con los valores de Similitud obtenidos En `NUMPY` a partir de las matrices nym con funciones. Se realiza la validación de los resultados obtenidos con la Matriz de Correlación de Pearson en Entumecido.' )
    st . markdown ( "<h2 style='text-align: ; color: black;'>Validación de Resultados de Pandas</h2>" , unsafe_allow_html = True )
    st . markdown ( 'Valores de Similitud en pandas' )
    pandas  =  pd . read_csv ( 'ENCUESTA.csv' )
    marco de datos1  =  pd . Marco de datos ( pandas )
    lista1  =  dataframe1 [ 'Dirección de correo electrónico' ]. tolista ()
    data['CREO EN TI']=data['CREO EN TI'].replace(np.nan,4)
    data['COLGANDO EN TUS MANOS']=data['COLGANDO EN TUS MANOS'].replace(np.nan,4)
    data['MAMIII']=data['MAMIII'].replace(np.nan,3)
    data['HEY MOR']=data['HEY MOR'].replace(np.nan,3)
    data['DANZA KUDURO']=data['DANZA KUDURO'].replace(np.nan,4)
    data['DESPECHA']=data['DESPECHA'].replace(np.nan,3)
    data['QUE MAS PUES ']=data['QUE MAS PUES '].replace(np.nan,3)
    data['UN VERANO SIN TI']=data['UN VERANO SIN TI'].replace(np.nan,3)
    data['NINACHAY']=data['NINACHAY'].replace(np.nan,5)
    data['POR USTEDES']=data['POR USTEDES'].replace(np.nan,4)
    data['MATADOR']=data['MATADOR'].replace(np.nan,3)
    data['MONTONERO AREQUIPENO']=data['MONTONERO AREQUIPENO'].replace(np.nan,4)
    data['MALA TU']=data['MALA TU'].replace(np.nan,3)
    data['CON LA BRISA ']=data['CON LA BRISA '].replace(np.nan,3)
    data['MY LIFE ']=data['MY LIFE '].replace(np.nan,4)
    data['SIGN OF THE TIMES ']=data['SIGN OF THE TIMES '].replace(np.nan,4)
    data['BLOODY MARY']=data['BLOODY MARY'].replace(np.nan,4)
    data['PROPUESTA INDECENTE']=data['PROPUESTA INDECENTE'].replace(np.nan,4)   
    n = data[data1.columns[1:]].to_numpy()
    m = data[data1.columns[0]].to_numpy()
    df = pd.DataFrame(n.T, columns = m)

    m_corr_pandas = df.corr()
    m_corr_pandas
    m_corr_p = np.round(m_corr_pandas, decimals = 4)
    
    con  st . eco ():
        
    st . write ( 'noeliaparedesgu@gmail.com y gparedesg@unsa.edu.pe obtienen el PRIMER indice mas alto de similitud')
    st . write ( 'noeliaparedesgu@gmail.com y elhuamani@unsa.edu.pe obtienen el SEGUNDO indice mas alto de similitud')


      
    
si  conclusiones :
    

    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )  
    st . markdown ( "<h2 style='text-align: center ; color: black;'>Conclusiones</h2>" , unsafe_allow_html = True )
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: ; color: black;'>¿Se valido o no los resultados?</h2>" , unsafe_allow_html = True )
    st . markdown ( 'Se realizo un formulario en google, para despues ser convertidos en un archivo CSV y utilizarlos conjuntamente con las librerias de data science Pandas, Seaborn, Matplotlib y Numpy. ' )
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: ; color: black;'>¿Es efectivo el método de conexión de pearson?</h2>" , unsafe_allow_html = True )
    st . markdown ( 'La correlacion de pearson fue eficiente para calcular la correlacion existente entre los datos que teniamos , estos valores necesitan ser de manera cuantitativa expresados ​​en valores numericos. Por los tanto podemos decir que el metodo de correlacion de Pearson si es efectivo.')
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: ; color: black;'>Los resultados Validados son:</h2>" , unsafe_allow_html = True )
    st . markdown ( 'Segun los resultados que fueron validos con Pandas y nuestro algoritmo los datos validados seria' )
    st . markdown ( '1. noeliaparedesgu@gmail.com y gparedesg@unsa.edu.pe obtienen el PRIMER indice mas alto de similitud' )
    st . markdown ( '2. noeliaparedesgu@gmail.com y elhuamani@unsa.edu.pe obtienen el SEGUNDO indice mas alto de similitud' )
    
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    st . markdown ( "<h2 style='text-align: ; color: black;'>Correlación de Pearson y Regresión Lineal, ¿cual es su relación?</h2>" , unsafe_allow_html = True )
    st . markdown ( 'La correlación de Pearson, es el que mide la regresión lineal de los datos obtenidos, por lo tanto la regresión lineal es parte de la correlación de pearson. Siendo la correlación de pearson el método y la regresión lineal el resultado.' )
    
    
    st . imagen ( 'https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png' )
    
    
