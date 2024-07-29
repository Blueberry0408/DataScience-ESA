#!/usr/bin/env python
# coding: utf-8

# # ESA 10 - Aaliyah Roderer

# ### Objective:
# To categorize the countries using socio-economic and health factors that determine the overall development of the country.
# 
# ### Problem Statement:
# HELP International has been able to raise around $ 10 million. Now the CEO of the NGO needs to decide how to use this money strategically and effectively.  Hence, your Job as a Data scientist is to categorize the countries using some socio-economic and health factors that determine the overall development of the country.

# In[26]:


import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/osual/OneDrive/Dokumente/Master/Data_Science/ESA10/country.txt')

df.info()
print(50*'-')


# In[27]:


df.describe()


# In[28]:


df.isnull().sum()


# In[29]:


df['country'].nunique()


# # Task:
# 
# ## 1. use K-means clustering and Hierarchical clustering to cluster the country into groups. please choose the number of the cluster in a rational reason

# In[30]:


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

os.environ['OMP_NUM_THREADS'] = '1'

df_clustering = df.drop('country', axis=1)

scaler = StandardScaler()
df_clustering_scaled = scaler.fit_transform(df_clustering)

# Silhouette-Score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(df_clustering_scaled)
    silhouette_avg = silhouette_score(df_clustering_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

optimal_clusters = 4

# K-Means Clustering
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(df_clustering_scaled)
df['KMeans_Cluster'] = kmeans_labels

# Hierarchical Clustering
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(df_clustering_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Countries')
plt.ylabel('Euclidean distances')
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters, affinity='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(df_clustering_scaled)
df['Hierarchical_Cluster'] = hierarchical_labels


# Die Anzahl der CLuster wurde basierend auf der Ausgabe des Sillhoutte Score gewähl und durch ausprobieren in der Visualisierung optimiert. Persönlich hatte ich das Gefühl, dass 4 Cluster den meisten Sinn machen, auch wenn die Elbow Methode beispielsweise eher auf 3 Cluster hingedeutet hat.

# ## 2. use PCA to reduce the dimension to 2d, and visualize the cluster from K-means and Hierarchical clustering respectively

# In[31]:


pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_clustering_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, cmap='viridis', label='KMeans Clusters')
plt.title('K-Means Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=hierarchical_labels, cmap='plasma', label='Hierarchical Clusters')
plt.title('Hierarchical Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()


# ## 3. please write the suggestion to CEO about the country you suggest

# Hier sind die Empfehlungen für die Verwendung der von HELP International gesammelten 10 Millionen Dollar:
# 
# Cluster 0: Dieser Cluster repräsentiert wahrscheinlich die Länder mit den größten sozioökonomischen und gesundheitlichen Problemen. Sie haben eine hohe Kindersterblichkeitsrate, eine niedrige Lebenserwartung und ein niedriges Pro-Kopf-BIP. Ein erheblicher Teil der Mittel sollte hier zur Deckung der grundlegenden Bedürfnisse in den Bereichen Gesundheit, Bildung und Infrastruktur eingesetzt werden.
# 
# Cluster 1: Die Länder in diesem Cluster stehen möglicherweise vor mäßigen Herausforderungen. Sie könnten von gezielten Maßnahmen wie der Verbesserung von Gesundheitseinrichtungen, der Verbesserung von Bildungsmöglichkeiten und der Unterstützung von Kleinunternehmen zur Förderung des Wirtschaftswachstums profitieren.
# 
# Cluster 2: Diese Länder befinden sich wahrscheinlich auf einem positiven Entwicklungspfad, haben aber immer noch verbesserungsbedürftige Bereiche. Hier könnten sich die Investitionen auf die Förderung der Technologie, die Verbesserung der Hochschulbildung und die Förderung von Innovationen zur Unterstützung des Wachstums konzentrieren.
# 
# Cluster 3: Die Länder in diesem Cluster dürften relativ wohlhabend sein und über bessere sozioökonomische Indikatoren verfügen. Hier könnten die Mittel zur Unterstützung von Forschung und Entwicklung, zur Förderung der ökologischen Nachhaltigkeit und zur Förderung globaler Partnerschaften eingesetzt werden.
# 
# Im Allgemeinen lässt sich also festhalten, dass die vorhandenen Gelder vor allem Dingen für Cluster 0 und 1 aufgewendet werden sollten, da es hier die größten Defizite zu vermerken gibt,

# ## Weitere eigene Ansichten und Tests

# In[ ]:


cluster0=kmeans_df[kmeans_df['KMeans_Clusters']==0]['country']
cluster1=kmeans_df[kmeans_df['KMeans_Clusters']==1]['country']
cluster2=kmeans_df[kmeans_df['KMeans_Clusters']==2]['country']
cluster3=kmeans_df[kmeans_df['KMeans_Clusters']==3]['country']


print("Number of countries in cluster 0",len(cluster0))
print("Number of countries in cluster 1",len(cluster1))
print("Number of countries in cluster 2",len(cluster2))
print("Number of countries in cluster 3",len(cluster3))


# In[ ]:





# In[60]:


def save_cluster_tables(df, cluster_col, method_name):
    unique_clusters = df[cluster_col].unique()
    for cluster in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster]
        cluster_data.to_csv(f'{method_name}_Cluster_{cluster}.csv', index=False)
        print(f'Data for {method_name} Cluster {cluster} saved to {method_name}_Cluster_{cluster}.csv')

save_cluster_tables(df, 'KMeans_Cluster', 'KMeans')

save_cluster_tables(df, 'Hierarchical_Cluster', 'Hierarchical')


# In[36]:


import pandas as pd
import os

directory = '.'  

csv_files = [
    'KMeans_Cluster_3.csv',
    'KMeans_Cluster_1.csv',
    'KMeans_Cluster_0.csv',
    'KMeans_Cluster_2.csv',
    'Hierarchical_Cluster_0.csv',
    'Hierarchical_Cluster_1.csv',
    'Hierarchical_Cluster_2.csv',
    'Hierarchical_Cluster_3.csv'
]

for file_name in csv_files:
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        print(f'Data for {file_name}:')
        df = pd.read_csv(file_path)
        print(df)
        print('---' * 10) 
    else:
        print(f'File {file_name} does not exist.')


# In[42]:


import pandas as pd
import os

directory = '.'  # Aktuelles Verzeichnis

csv_files = [
    'KMeans_Cluster_3.csv',
    'KMeans_Cluster_1.csv',
    'KMeans_Cluster_0.csv',
    'KMeans_Cluster_2.csv',
    'Hierarchical_Cluster_0.csv',
    'Hierarchical_Cluster_1.csv',
    'Hierarchical_Cluster_2.csv',
    'Hierarchical_Cluster_3.csv'
]

for file_name in csv_files:
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        variable_name = os.path.splitext(file_name)[0]
        df = pd.read_csv(file_path)
        globals()[variable_name] = df
        print(f'Data for {file_name} loaded into variable {variable_name}')
        print(df)
        print('---' * 10)
    else:
        print(f'File {file_name} does not exist.')


# In[44]:


KMeans_Cluster_0.describe()


# In[45]:


KMeans_Cluster_1.describe()


# In[46]:


KMeans_Cluster_2.describe()


# In[47]:


KMeans_Cluster_3.describe()


# In[48]:


Hierarchical_Cluster_0.describe()


# In[49]:


Hierarchical_Cluster_1.describe()


# In[50]:


Hierarchical_Cluster_2.describe()


# In[51]:


Hierarchical_Cluster_3.describe()


# In[ ]:





# In[ ]:




