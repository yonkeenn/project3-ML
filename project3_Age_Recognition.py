#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############################################################
####################### Librerias usadas  ###################
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from pydub import AudioSegment
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pickle
#############################################################


# In[6]:


#############################################################
################### Preparacion de Datos  ###################

# Crear expectogramas a partir de los audios.
# 1. Convertir los audios.wav en png para poder crear los atributos.
   
# Definimos el color map que usaremos para crear el espectograma
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))

# ./TestData es la ruta donde esta todos los audios etiquetados Audio00x_xyz_sloganw.
path = r'./RawData'
extension = '.wav'


for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            name=os.path.splitext(file_name)[0]
            y, sr = librosa.load(file_name_path, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'{file_name_path[:-3]}png')
            plt.clf()
#############################################################


# In[52]:


#############################################################
################### Creacion del dataset  ###################

# Funcion para hacer el mapping de los archivos de audio con la clase
def identificar_parametros(filename):
    cadena1 = file_name
    if cadena1.find("otros")>=0 :
        return "otros"       
    if cadena1.find("rango1")>=0 :
        return "rango1"
    if cadena1.find("rango2")>=0 :
        return "rango2"
    if cadena1.find("rango3")>=0 :
        return "rango3"
    else:
        return 999

# Creamos la lista de clases
slogan = 'rango1 rango2 rango3 otros'.split()

# Creamos la cabecera del data set
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' clase'
header = header.split()

# Creamos el archivo dataset.csv que tiene como primera columna la variale header
file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

# Extraemos las caracteristicas desde el Espectograma que se convertiran en atributos del dataset: 
# 1. Mel-frequency cepstral coefficients (MFCC)
# 2. Spectral Centroid
# 3. Zero Crossing Rate
# 4. Chroma Frequencies
# 5. Spectral Roll-off.
path = r'./RawData'
extension = '.wav'
for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            audio = file_name_path
            y, sr = librosa.load(audio, mono=True, duration=30)
            rmse = librosa.feature.rms(y=y)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            p_slogan = identificar_parametros(file_name)
            to_append = f'{file_name_path} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {p_slogan}'
            file = open('dataset.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

# Cargamos y leemos los datos en la variable pandas data.
# Contiene las etiquetas y atributos.
data = pd.read_csv('dataset.csv')
data.head(10)


# In[53]:


###################################################################################################
############## Preparacion, Entendimiento, Limpieza y Normalizacion del dataset  ###################

###### 1. NULLs or NA #######
# Verificamos si existen datos Null, NA en el dataset.
data = pd.read_csv('dataset.csv')
print("Datos Null y NA en el dataset")
print("============================= \n")
print(data.isnull().sum())
print("\n")
print("Datos Null en todo el dataset: ", data.isnull().values.any())
print("Datos NA en todo el dataset: ", data.isna().sum().sum())

###### 2. Outlayers or Datos extremos #######
# Verificamos si existen datos extremos en el dataset.
data = pd.read_csv('dataset.csv')

print(data["clase"].value_counts())

print("Asimetria de los datos:\n", data.skew())

#Filtramos las columanas para poder analizar por grupos
columns = data.columns
print(data.columns)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
cols = [c for c in data.columns if data[c].dtype in numerics]

cols1 = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'] 
cols2 = ['mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10']
cols3 = ['mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']

#Generamos 3 tipos de dataset para entender la dispercion que tienen los datos
data1 = data[cols1]
data2 = data[cols2]
data3 = data[cols3]

ax1 = data1.hist(bins= 10, alpha=0.5)
ax1 = data2.hist(bins= 10, alpha=0.5)
ax1 = data3.hist(bins= 10, alpha=0.5)
plt.show()



# Calculamos los quartiles y el IQR
Q0 = data1.quantile(0.00)
Q1 = data1.quantile(0.25)
Q2 = data1.quantile(0.50)
Q3 = data1.quantile(0.75)
Q4 = data1.quantile(1.00)

# Creamos el IRQ con el 3er y 1er quartil
IQR= Q3 - Q1

# Creamos el filtro que nos validara la existencia de valores extremos que este fuera del 1.5*IRQ en cada columna
filtro = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
outliers = data[filtro]

# Sacamos los datos extemos y actualizamos el dataset
data_new = data[~filtro]

# Actualizamos dataset
data = data_new

#Grafico de los datos extremos
plt.figure(figsize=(20,20))
outliers.boxplot()
plt.show()

#Grafica sin los datos extremos
plt.figure(figsize=(20,20))
data_new.boxplot()
plt.show()

data.head()

###### 4. Eliminacion de columnas innesesarias #######
# Eliminamos las columanas que no utilizamos como la columna "filename" que es usada como nombre del archivo.
data = data.drop(['filename'],axis=1)
data.head(10)

###### 5. Normalizacion #######

X = data.iloc[:, :-1]
print(np.array(data.iloc[:, :-1]))
      
minmax_scale = preprocessing.MinMaxScaler().fit(X)
X = minmax_scale.transform(np.array(data.iloc[:, :-1], dtype = float))

############ 6. Codificacion de la clase ##############
      
clase_list = data.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(clase_list)
data["clase"] = y
print(y)

data.head(10)


# In[50]:



###### 3. Balanceo de datos mediante las clases #######
# Primero verificamos que una gran diferencia entre la clase "otros" y las otras clases slogan1,2,3,4,5...

# Buscamos los valores de cada clase en base a filas
rango3, rango2, rango1 = data["clase"].value_counts()

# Segun esto podemos verificar que existe un gran desbalance entre la clase "otros" y las otras clases
# Cantidad de filas por tipo de clase
print("Cantidad de filas por Clase antes del balanceo: \n", data["clase"].value_counts())   

#Creamos un filtro para cada clase
g1 = data.loc[data["clase"]=="rango1"]
g2 = data.loc[data["clase"]=="rango2"]
g3 = data.loc[data["clase"]=="rango3"]

# Luego aplicamos la tecnica de Over-sample, que es utilizada para rellenar a las cantidades minoritarias
# que en este caso son las clases slogan1, slogan2, ... para que se iguale en cantidad con la clase otros.
g1_over = g1.sample(rango3, replace=True)
g2_over = g2.sample(rango3, replace=True)
g3_over = g3.sample(rango3, replace=True)


# Balanceamos los datos
data_balanced = pd.concat([g1_over, g2_over, g3_over])
# Volvemos a verificar el desbalance entre la clase "otros" y las otras clases, y confirmamos que ya no existe 
# el desbalance.
# Cantidad de filas por tipo de clase luego del balanceo
print("\n")
print("Cantidad de filas por Clase despues del balanceo: \n", data_balanced['clase'].value_counts())
print(data_balanced.shape)
data_balanced.head(10)

# Actualizamos dataset
data = data_balanced

###### 4. Eliminacion de columnas innesesarias #######
# Eliminamos las columanas que no utilizamos como la columna "filename" que es usada como nombre del archivo.
data = data.drop(['filename'],axis=1)
data.head(10)

###### 5. Normalizacion #######

X = data.iloc[:, :-1]
print(np.array(data.iloc[:, :-1]))
      
minmax_scale = preprocessing.MinMaxScaler().fit(X)
X = minmax_scale.transform(np.array(data.iloc[:, :-1], dtype = float))

############ 6. Codificacion de la clase ##############
      
clase_list = data.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(clase_list)
data["clase"] = y
print(y)

data.head(10)


# In[55]:


#############################################################
######### Creacion del Modelo y Entrenamiento ###############
#
# Para crear el modelo usaremos la tecnica Stratified K-fold (version mejorada de la tecnica K-fold) 
#para separar con la data de prueba y de entrenamiento    

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

def isnp(M):
	if isinstance(M, np.ndarray): return True
	return False

def dim(M):
    if isnp(M): M = list(M)

    if not type(M) == list:
        return []

    return [len(M)] + dim(M[0])


# Creacion del modelo con el algoritmo MLPClassifier ANN con las siguientes caracteristicas
# 1. Uso funcion de activacion softmax en la ultima capa(Esta configuracion esta en la logica del algoritmo)
# 2. Uso maximo de 300 iteracion 
# 3. Uso de 3 capas
clf = MLPClassifier(random_state=1, max_iter=300)

#######
# separa los datos para entrenamiento y para pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.2)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# entrena clasificador
clf = MLPClassifier(random_state=1, max_iter=500).fit(X_train,y_train)
print(clf)

# calcula probabilidades, solo el primero
y_test_prob = clf.predict_proba(X_test)
#print('X_test', X_test)
#print('y_test_prob', y_test_prob)

# predice resultados, solo el primero
y_test_pred = clf.predict(X_test)
#print('X_test', X_test)
#print('y_test_prd', y_test_pred)

# matriz de confusion
cmatrix = confusion_matrix(y_test, y_test_pred)
print(cmatrix)
plot_confusion_matrix(clf, X_test, y_test)
plt.show()

# Evaluacion del modelo
print("Score: ",  clf.score(X_test, y_test))
print("Acurracy: ", accuracy_score(y_test,y_test_pred)*100)
print("Recall: ", recall_score(y_test, y_test_pred, average=None))
print("Precision: ", precision_score(y_test, y_test_pred, average=None)*100)

print("Funcion de acctivacion en la ultima capa: ", clf.out_activation_)
print("Perdida obtenida durante el entranamiento")
print(clf.loss_)
print(clf.best_loss_)
print("Numero iteraciones")
print(clf.n_iter_)
print("Numero de capas")
print(dim(clf.coefs_))

#######

pd.DataFrame(clf.loss_curve_).plot()
plt.show()


# In[62]:


#############################################################
################### Evaluacion del Modelo ###################

# Creacion del clasificador
pickle.dump(clf, open('classifier.pkl', 'wb'), protocol=4)

# Cargamos el modelo para su evaluacion
clf = pickle.load(open('classifier.pkl', 'rb'))


label = {0: 'rango1', 
         1: 'rango2', 
         2: 'rango3'
        }

# Creacion de dataset(evaluación.csv) sin clase el cual se usara pra evaluar el modelo.

cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))

path = r'./Test'
extension = '.wav'
for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            name=os.path.splitext(file_name)[0]      
            y, sr = librosa.load(file_name_path, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'{file_name_path[:-3]}png')
            plt.clf()
            
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header = header.split()

# Extraer caracteristicas desde el Espectograma: Mel-frequency cepstral coefficients (MFCC), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, y Spectral Roll-off.
file = open('evaluacion.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
    
extension = '.wav'
for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            audio = file_name_path
            y, sr = librosa.load(audio, mono=True, duration=30)
            rmse = librosa.feature.rms(y=y)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{file_name_path} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            file = open('evaluacion.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

# Procesamiento de datos: Cargar la data del CSV, codificacion de etiquetas, división de datos en conjunto de entrenamiento y prueba.
data = pd.read_csv('evaluacion.csv')
print(data.columns)
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

print("Evaluacion del Modelo con la data de prueba: ")
print("============================================ \n")
for i in range(0,len(X)):
    print('Prediction %s Probability: %.2f%%' %          (label[clf.predict(X)[i]], clf.predict_proba(X)[i].max()*100))


# In[66]:





# In[ ]:




