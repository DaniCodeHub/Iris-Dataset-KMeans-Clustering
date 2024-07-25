#Import initial modules and libraries

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = load_iris(return_X_y=False, as_frame=False)
X = data.data
y = data.target
target_names = data.target_names
feature_names = data.feature_names

#Make a pandas dataframe
df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
df['species'] = target_names[y]

#Input parameters into KMeans estimator and fit to X values 'data'
KMean_model = KMeans(n_clusters=3, random_state=1000)
KMean_model.fit(X)


#Make predictions/labels_ with the KMeans fitted model
predictions = KMean_model.predict(X)

labels = KMean_model.labels_


#Measure results with silhouette_score and inetria metrics
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(X, labels)
print(silhouette_avg)

KMean_model.inertia_

#Lets Use elbow method to determine the optimal number of peaks. Append corresponding inertia values for number of clusters and append to list 'wcss'
wcss = []
for K in range(1,11):
    model = KMeans(n_clusters=K, random_state=1000)
    model.fit(X)
    wcss.append(model.inertia_)

print(wcss)

#Plot the  inertia values against number of clusters to determine the elbow and see what quantity of clusters is most optimal
plt.plot(range(1,11), wcss)

plt.xticks(range(1, 11))
plt.axvline(x=3, color='red', linestyle='--', linewidth=1, label='Inflection Point: Clusters = 3')


plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')

plt.legend()

plt.show()

#Plot the first two feautures 'sepal length' and 'sepal width' from the Iris dataset and plot the centroids for these features.
#The results should account for all the rows including the three distinct target species.

clusters = KMean_model.cluster_centers_

from matplotlib.patches import Patch

plt.figure(figsize=(15, 10))

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(clusters[:, 0], clusters[:, 1], c='red', marker='x')

plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris Dataset KMeans clustering')

plt.colorbar(label='Cluster Label')

legend_elements = [Patch(facecolor='#E6D369', label='setosa'),
                   Patch(facecolor='#29A6A6', label='virginica'),
                   Patch(facecolor='purple', label='versicolor'),
                   Patch(facecolor='red', label='centroid')
]

plt.legend(handles=legend_elements)

plt.show()

