# Face-Detection--OPenCV-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

X,y=make_moons(n_samples=300,noise=0.05,random_state=42)
spectral=SpectralClustering(n_clusters=2,affinity='nearest_neighbors',random_state=42)
labels=spectral.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis')
plt.title("Graph Based Clustering (Spectral Clustering )")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
