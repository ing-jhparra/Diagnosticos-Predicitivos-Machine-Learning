import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  classification_report
import warnings
warnings.simplefilter("ignore")

dataset = r'.\Datasets\BBDD_Hospitalización.xlsx'

try:
    df = pd.read_excel(dataset)
    print(f'Ha sido cargado dataset {dataset}')
except Exception as e:
    print(f'Error al cargar el dataset {e}')

columnas_a_excluir = ["HOSPITALIZACIÓN ULTIMO MES",
                     "BIOPSIAS PREVIAS",
                     "VOLUMEN PROSTATICO",
                     "ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS",
                     "NUMERO DE MUESTRAS TOMADAS",
                     "NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA",
                     "TIPO DE CULTIVO",
                     "PATRON DE RESISTENCIA",
                     "DIAS HOSPITALIZACION MQ",
                     "DIAS HOSPITALIZACIÓN UPC"]

df_hosp = df.drop(columnas_a_excluir, axis=1)
print(f'Excluyendo variables {columnas_a_excluir}')
print('Mostrando estadisticas de variables estadisiticas')
print(round(df_hosp.describe(),2))

df_hosp.loc[df_hosp["EDAD"] > 84, "EDAD"] = df_hosp["EDAD"].median()
print('Se procesan valores atipicos en Edad')

valor = df_hosp["HOSPITALIZACION"].value_counts()
if valor[0] > valor[1]:
    imputacion = 'NO'
else: 
    imputacion = 'SI'
df_hosp["HOSPITALIZACION"].fillna(value=imputacion, inplace=True)
print('Se procesan valores nulos en la variable objetivo (y)')

df_hosp.drop([568,569], axis=0, inplace=True)
print('Eliminacion de filas con valores nulos')

df_hosp["PSA"].fillna(value=df_hosp["PSA"].median(), inplace=True)
print('Se procesa imputacion de valores nulos en la variable PSA')

valor = df_hosp["CUP"].value_counts()
if valor[0] > valor[1]:
    imputacion = 'NO'
else: 
    imputacion = 'SI'
df_hosp["CUP"].fillna(value=imputacion, inplace=True)
print('Se procesan valores nulo en variable CUP')

valor = df_hosp["ENF. CRONICA PULMONAR OBSTRUCTIVA"].value_counts()
if valor[0] > valor[1]:
    imputacion = 'NO'
else: 
    imputacion = 'SI'
df_hosp["ENF. CRONICA PULMONAR OBSTRUCTIVA"].fillna(value=imputacion, inplace=True)
print('Se procesan valores nulo en variable ENF. CRONICA PULMONAR OBSTRUCTIVA')

valor = df_hosp["AGENTE AISLADO"].value_counts()
if valor[0] > valor[1]:
    imputacion = 'NO'
else: 
    imputacion = 'SI'
df_hosp["AGENTE AISLADO"].fillna(value=imputacion, inplace=True)
print('Se procesan valores nulo en variable AGENTE AISLADO')

columnas_a_renombrar = {
    "EDAD" : "edad",
    "DIABETES" : "diabetes",
    "PSA" : "psa",
    "CUP" : "cup",
    "ENF. CRONICA PULMONAR OBSTRUCTIVA" : "enf_pulmonar",
    "BIOPSIA" : "biopsia",
    "FIEBRE" : "fiebre",
    "ITU" : "itu",
    "AGENTE AISLADO" : "agente_aislado",
    "HOSPITALIZACION" : "hospitalizacion"
}

df_hosp.rename(columns = columnas_a_renombrar,inplace=True)
print(f'Se renombra variables {columnas_a_renombrar}')

enfermedad_pulmonar_cronica = {'SI, ASMA': 'SI','SI, EPOC': 'SI',}
df_hosp["enf_pulmonar"].replace(enfermedad_pulmonar_cronica,inplace=True)
print('Se realiza transformacion en la variable enf_pulmonar')

biopsia = {'NEG': 'NEGATIVO',
           'ADENOCARCINOMA GLEASON 6 ': 'ADENOCARCINOMA GLEASON 6',
           'ADENOCARCINOMA GLEASON 7 ': 'ADENOCARCINOMA GLEASON 7',
           'ADENOCARCINOMA GLEASON 10 ': 'ADENOCARCINOMA GLEASON 10',
           'ADENOCARCINOMA GLEASON 9 ': 'ADENOCARCINOMA GLEASON 9',
           'ADENOCARCINOMA GLEASON 8 ': 'ADENOCARCINOMA GLEASON 8'
          }
df_hosp["biopsia"].replace(biopsia,inplace=True)
print('Se realiza transformacion en la variable biopsia')

cantidad = df_hosp.duplicated().sum()
df_hosp.drop_duplicates(inplace=True)
print(f'Se eliminan {cantidad} valores duplicados')

reemplazar_valores = dict(NO=0,SI=1) 
df_hosp["diabetes"].replace(reemplazar_valores,inplace=True)
df_hosp["cup"].replace(reemplazar_valores,inplace=True)
df_hosp["enf_pulmonar"].replace(reemplazar_valores,inplace=True)
df_hosp["fiebre"].replace(reemplazar_valores,inplace=True)
df_hosp["itu"].replace(reemplazar_valores,inplace=True)
df_hosp["hospitalizacion"].replace(reemplazar_valores,inplace=True)
print('Se realiza transformacion en las variables diabetes, cup, enf_pulmonar, fibre, itu, hospitalizacion')

biopsia_dummies = pd.get_dummies(df_hosp["biopsia"],prefix="Biopsia",prefix_sep="_",dtype=int)
agente_dummies = pd.get_dummies(df_hosp["agente_aislado"],prefix="Agente",prefix_sep="_",dtype=int)
enfermedad_pulmonar_dummies = pd.get_dummies(df_hosp["enf_pulmonar"],prefix="Enf_Pulmonar",prefix_sep="_",dtype=int)
hospitalizacion = pd.concat([df_hosp,agente_dummies,biopsia_dummies,enfermedad_pulmonar_dummies], axis=1)
hospitalizacion.drop(["biopsia","agente_aislado","enf_pulmonar"],axis="columns",inplace=True)
print('Se realiza transformaciones a dummies en biopsia, agente y enf_pulmonar')

hospitalizacion.to_csv(r'.\Datasets\BDHospitalizacion.csv', index=False)
print('Se ha almacenado una base de datos con las transformaciones necesaria y lista para ser utilizado en el modelo')

dataset = r'.\Datasets\BDHospitalizacion.csv'

try:
    df= pd.read_csv(dataset)
    print(f'Ha sido cargado dataset {dataset}')
except Exception as e:
    print(f'Error al cargar el dataset {e}')

y = df["hospitalizacion"]
X = df.drop("hospitalizacion", axis=1)
print('Se ha creado el conjunto de datos X y y')

clases = pd.value_counts(y,sort=True)
if clases[0] < clases[1]:
    tasa = round(clases[0] / (clases[0] + clases[1]) * 100,2)
    mensaje = f'La clase "NO" representa un {tasa}% del total de muestras'
elif clases[1] < clases[0]:
    tasa = round(clases[1] / (clases[0] + clases[1]) * 100,2)
    mensaje = f'La clase "SI" representa un {tasa}% del total de muestras'
else :
    print('Verificar ...')
print(mensaje)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=19) # Dividimos el dataset en entrenamiento y test
print(f'Datos para el entrenamiento. Dimension de X_train : {X_train.shape}, Dimension para y_train : {y_train.shape}')
print(f'Datos para el test. Dimension de X_test : {X_test.shape}, Dimension para y_test : {y_test.shape}')

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(class_weight={1:22.7}, # Para el valor de class_weight, el 1 es la etiqueta Si, y el 22.7 sale de el total de No entre el total de Si = 544/24 = 22.7
                             criterion = 'gini', 
                             random_state = 40,
                             max_depth = 4) # Crear un clasificador

clf.fit(X_train, y_train) # Entrene el modelo de clasificación de árbol de decisión en el conjunto de entrenamiento.# Entrenamos
y_pred = clf.predict(X_test) # predicciones de los datos de prueba X_test con el modelo entrenado
print('Se ha instanciado y entrenado el modelo arbol de decision')

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
cm = confusion_matrix(y_test,y_pred)

print('\nMatriz de Confusion del modelo')
print(cm)

print('\nDatos estadisticos extraidos de la matriz de confusion')
target_names = ['No', 'Si']
print(classification_report(y_test, y_pred,target_names=target_names))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'max_depth': randint(low=8,high=50)
    }

clf = DecisionTreeClassifier(class_weight={1:22.7}, # Para el valor de class_weight, el 1 es la etiqueta Si, y el 22.7 sale de el total de No entre el total de Si = 544/24 = 22.7
                             criterion = 'gini', 
                             random_state = 40)

busqueda_vc = RandomizedSearchCV(clf, param_distributions = param_distribs, n_iter=8, cv=3, scoring='r2')
busqueda_vc.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
cm = confusion_matrix(y_test,y_pred)
print('\nMatriz de Confusion del modelo luego de ajustar hiperparametros')
print(cm)
print('\nDatos estadisticos extraidos de la matriz de confusion')
target_names = ['No', 'Si']
print(classification_report(y_test, y_pred,target_names=target_names))

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # p es el parámetro de la distancia de Minkowski 
                                                                                #(p = 2 es la distancia euclidiana, p=1 es la distancia de Manhattan)
                                                                                #metric = 'minkowski' es la distancia de Minkowski
classifier.fit(X_train, y_train)
print('Se instancio y se entreno con el algoritmo de vecinos mas cercanos')

y_pred = classifier.predict(X_test) # predecir los valores de X_test con el modelo entrenado (classifier)

print('\nDatos estadisticos extraidos de la matriz de confusion para el algoritmo de vecinos mas cercanos')
target_names = ['No', 'Si']
print(classification_report(y_test, y_pred,target_names=target_names))

print('Hasta luego ... :)')