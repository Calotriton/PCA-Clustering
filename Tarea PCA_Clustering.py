# -*- coding: utf-8 -*-

# PREPARACIÓN DE LOS DATOS #
# Establecer tu directorio de trabajo
import os
#os.chdir(r'directorio')
#print("Directorio de trabajo actual:", os.getcwd())

# Importamos las librerías que se usarán
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from FuncionesMineria2 import (plot_varianza_explicada, plot_cos2_heatmap, plot_corr_cos, plot_cos2_bars,
                               plot_contribuciones_proporcionales, plot_pca_scatter, plot_pca_scatter_with_vectors,
                               plot_pca_scatter_with_categories)

# Cargamos el dataset de la tarea en un DataFrame
df = sns.load_dataset('penguins')

# Vemos las primeras filas del df
df.head()

# Creamos una lista con los nombres de las variables
variables = list(df)
variables

# Creamos otra lista con las variables numéricas
vars = (variables[2], variables[3], variables[4], variables[5])
var_num = list(vars)


# ANÁLISIS SIMPLES EXPLORATORIOS #
# Mostramos los estadi­sticos descriptivos básicos de las variables numericas, creando un df con los resultados
estadisticos = pd.DataFrame({
    'Mí­nimo': df[var_num].min(),
    'Q1': df[var_num].quantile(0.25),
    'Mediana': df[var_num].median(),
    'Q3': df[var_num].quantile(0.75),
    'Media': df[var_num].mean(),
    'Máximo': df[var_num].max(),
    'Desv. tí­pica': df[var_num].std(),
    'Varianza': df[var_num].var(),
    'Coef. de Variación': (df[var_num].std() / df[var_num].mean()),
    'Missing': df[var_num].isna().sum()
    })

#Creamos un gráfico de dispersión para ver la estructura de los datos
sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species', palette='deep')
plt.title('Relacion entre la longitud y la profundidad del pico por especie')
plt.show()
plt.close()

# Creamos un violinplot para ver la distribución del peso de cada especie
sns.violinplot(x='species', y='body_mass_g', data=df, inner="quart")
plt.title('Distribucion de la Masa Corporal por Especie')
plt.show()
plt.close()


# ANÁLISIS DE COMPONENTES PRINCIPALES #
# Creamos un dataframe con solo las variables numéricas.
# Primero, eliminamos las observaciones con valores NaN
df = df.dropna()
df_num = df[var_num]

# Verificamos las primeras filas 
print(df_num.head())

# Cálculo y representación de la matriz de correlación entre las variables numéricas
R = df_num.corr()
print("Matriz de correlación entre variables numéricas:")
print(R)

# Representamos la matriz de correlación como un heatmap
sns.heatmap(R, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación entre Variables Numéricas')
plt.show()
plt.close()

#Seguidamente preparamos los datos para realizar el ACP
# Primero, estandarizamos los datos, utilizando tandardScaler() para estandarizar (normalizar) las variables, y creamos un df
df_std = pd.DataFrame(
    StandardScaler().fit_transform(df_num),  
    columns=['{}_z'.format(var_num) for var_num in var_num],  
    index=df_num.index  
)
df_std.head()

# Crea una instancia de Análisis de Componentes Principales (ACP) con 4 componentes máximo (número de variables)
pca = PCA(n_components=4)

# Aplicamos el Análisis de Componentes Principales (ACP) a los datos estandarizados:
fit = pca.fit(df_std)

# Obtenemos los autovalores asociados a cada componente principal
autovalores = fit.explained_variance_
autovalores

# Obtenemos la varianza explicada por cada componente principal como un porcentaje de la varianza total.
var_explicada = fit.explained_variance_ratio_

# Calcular la varianza explicada acumulada a medida que se agrega cada componente principal
var_acumulada = np.cumsum(var_explicada)
var_acumulada

# Creamos un dataframe con los datos anteriores y establecemos el índice
data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)])

# Imprimimos la tabla
print(tabla)

# Representacion de la variabilidad explicada:
plot_varianza_explicada(var_explicada, fit.n_components_)

# Seleccionamos pues dos componentes principales, por lo que se crea una nueva instancia de ACP y se aplica a los datos
pca = PCA(n_components=2)
fit = pca.fit(df_std)

# Obtenemos los autovectores asociados a cada componente principal.
print(pca.components_)

# Obtenemos los autovectores asociados a cada componente principal y se transponen
autovectores = pd.DataFrame(pca.components_.T,
                            columns = ['Autovector {}'.format(i) for i in range(1, fit.n_components_+1)],
                            index = ['{}_z'.format(var_num) for var_num in var_num])

display(autovectores)

# Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(df_std),
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=df_std.index)

# Añadimos las componentes principales a la base de datos estandarizada.
penguins_z_cp = pd.concat([df_std, resultados_pca], axis=1)
display(penguins_z_cp)

# Guardamos el nombre de las variables del archivo conjunto (variables y componentes).
variables_cp = penguins_z_cp.columns

# Calculamos las correlaciones y seleccionamos las que nos interesan (variables contra componentes).
correlacion = pd.DataFrame(np.corrcoef(df_std.T, resultados_pca.T),
                           index = variables_cp, columns = variables_cp)

n_variables = fit.n_features_in_
correlaciones_penguins_con_cp = correlacion.iloc[:fit.n_features_in_, fit.n_features_in_:]
display(correlaciones_penguins_con_cp)

cos2 = correlaciones_penguins_con_cp**2
display(cos2)

# Cramos el gráfico del cuadrado de las cargas en los componentes principales
plot_cos2_heatmap(cos2)

# Y el gráfico de la varianza explicada por cada variable de las componentes principales
plot_cos2_bars(cos2)

# Finalmente graficamos el resultado de ACP final con las coordenadas
plot_pca_scatter(pca, df_std, fit.n_components)


# Creamos ahora otro ACP pero con las categorías de especie para ver cuales representan
plot_pca_scatter_with_categories(df, resultados_pca.values, fit.n_components, 'species')


# Para terminar construimos un indice para valorar las caraceristicas conjuntas de un pingüino
# Usamos las cargas del primer componente principal 
cargas_pc1 = pca.components_[0]

# Calculamos el índice como la combinación lineal de las variables estandarizadas y las cargas de PC1
df['Indice_Fisico'] = np.dot(df_std.values, cargas_pc1)

# Mostramos los valores medios del índice para cada especie
indice_por_especie = df.groupby('species')['Indice_Fisico'].mean()

print("Índice medio para cada especie:")
print(indice_por_especie)

#Calculamos los índices específicos:
adelie_index = df[df['species'] == 'Adelie']['Indice_Fisico'].iloc[0]
chinstrap_index = df[df['species'] == 'Chinstrap']['Indice_Fisico'].iloc[0]
gentoo_index = df[df['species'] == 'Gentoo']['Indice_Fisico'].iloc[0]

print(f"\nÍndice para un pingüino 'Adelie': {adelie_index}")
print(f"Índice para un pingüino 'Chinstrap': {chinstrap_index}")
print(f"Índice para un pingüino 'Gentoo': {gentoo_index}")

# CLUSTERING JERÁRQUICO #
# Seleccionamos solo las columnas numéricas de df
df_num = df.select_dtypes(include=[np.number])

# Observamos la matriz mediante un Heatmap para hacernos una idea general
sns.clustermap(df_num, cmap='coolwarm', annot=True)
plt.show()

# Calculamos la matriz de distancias Euclideas
distance_matrix = distance.cdist(df_num, df_num, 'euclidean')

distance_small = distance_matrix[:5, :5]
distance_small = pd.DataFrame(distance_small, index=df_num.index[:5], columns=df_num.index[:5])
distance_small_rounded = distance_small.round(2)
print(distance_small_rounded)
 
# Comprobamos que esté correcto
df[:2]

# Graficamos
plt.figure(figsize=(8, 6))
df_distance = pd.DataFrame(distance_matrix, index = df_num.index, columns = df_num.index)
sns.heatmap(df_distance, annot=False, cmap="YlGnBu", fmt=".1f")
plt.show()

# Generamos el clustering jerárquico 
linkage = sns.clustermap(df_distance, cmap="YlGnBu", fmt=".1f", annot=False, method='average').dendrogram_row.linkage

# Reordenamos los datos según el clústering jerárquico obtenido
order = pd.DataFrame(linkage, columns=['cluster_1', 'cluster_2', 'distance', 'new_count']).index
reordered_data = df.reindex(index=order, columns=order)

# Añadimos una barra de color
sns.heatmap(reordered_data, cmap="YlGnBu", fmt=".1f", cbar=False)
plt.show()

# Estandarizamos las columnas del df
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)
print(df_std)

# Volvemos a calcular las distancias Euclideas con los datos estandarizados 
distance_std = distance.cdist(df_std, df_std,"euclidean")

print(distance_std[:5, :5].round(2))

# Graficamos
plt.figure(figsize=(8, 6))
df_std_distance = pd.DataFrame(
    distance_std, index=df_std.index, columns=df.index)
sns.heatmap(df_std_distance, annot=False, cmap="YlGnBu", fmt=".1f")
plt.show()

# Generamos el clustering jerárquico de nuevo, pero con los datos estandarizados
linkage = sns.clustermap(df_std_distance, cmap="YlGnBu", fmt=".1f",
                         annot=False, method='average').dendrogram_row.linkage

# Reordenamos los datos finales según el clústering jerárquico obtenido
order = pd.DataFrame(
    linkage, columns=['cluster_1', 'cluster_2', 'distance', 'new_count']).index
reordered_data = df_std.reindex(index=order, columns=order)

# Añadimos barra de color
sns.heatmap(reordered_data, cmap="YlGnBu", fmt=".1f", cbar=False)
plt.show()

# Calculamos la matriz
linkage_matrix = sch.linkage(df_std_distance, method='ward')

# Y creamos el dendrograma
dendrogram = sch.dendrogram(
    linkage_matrix, labels=df.index, leaf_font_size=9, leaf_rotation=90)

plt.show()

# Seleccionamos 3 grupos según los resultados del clústering jerárquico

# Asignamos los puntos de los datos a 3 clústers
num_clusters = 3
cluster_assignments = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Vemos como se añade cada observación a un grupo
print("Cluster Assignments:", cluster_assignments)

# Creamos la nueva variable 'Cluster' y le asignamos los valores de  'cluster_assignments' 
df['Cluster4'] = cluster_assignments

# Realizamos el ACP
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_std)

# Se crea un nuevo DF para los dos componentes principales
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Creamos el scatterplot
plt.figure(figsize=(10, 6))

# Añadimos cada punto a su clúster con el mismo color 
for cluster in np.unique(cluster_assignments):
    plt.scatter(df_pca.loc[cluster_assignments == cluster, 'PC1'],
                df_pca.loc[cluster_assignments == cluster, 'PC2'],
                label=f'Cluster {cluster}')

# Se añaden etiquetas
for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(df.index[i]), fontsize=8)

# Ploteamos
plt.title("2D PCA Plot with Cluster Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# CLUSTERING NO JERÁRQUICO #
# Seleccionamos el número de clústers (k=3)
k = 4

# Iniciamos el modelo k-means
kmeans = KMeans(n_clusters=k, random_state=0)

# Usamos los datos estandarizados previos en el modelo k-means
kmeans.fit(df_std)

# Ponemos las etiquetas del clúster en los datos 
kmeans_cluster_labels = kmeans.labels_

print(kmeans_cluster_labels)

# Repetimos el gráfico anterior con los resultados del k-means
plt.figure(figsize=(10, 6))

for cluster in np.unique(kmeans_cluster_labels):
    plt.scatter(df_pca.loc[kmeans_cluster_labels == cluster, 'PC1'],
                df_pca.loc[kmeans_cluster_labels == cluster, 'PC2'],
                label=f'Cluster {cluster}')

for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(df.index[i]), fontsize=8)

plt.title("2D PCA Plot with K-means Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# Creamos un array para almacenar los distintos valores de  k.
wcss = [] 

for k in range(1, 11):  
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)  #

# Realizamos el plot del método del codo para determinar el número óptimo de clústers
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Usamos el método de la silueta para comparar la misma comprobación
# Creamos un array para almacenar los distintos valores de  k.

silhouette_scores = []

# Realizamos un clústering k-means y calculamos las siluetas 

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_std, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Calculamos el clustering con el numero optimo obtenido antes (k=3) y etiquetamos cada valor
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df_std)
labels = kmeans.labels_

silhouette_values = silhouette_samples(df_std, labels)
silhouette_values

#Representamos las siluetas 
plt.figure(figsize=(8, 6))
y_lower = 10

for i in range(3):
    ith_cluster_silhouette_values = silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.get_cmap("Spectral")(float(i) / 3)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10


plt.title("Silhouette Plot for Clusters")
plt.xlabel("Silhouette Coefficient Values")
plt.ylabel("Cluster Label")
plt.grid(True)
plt.show()


