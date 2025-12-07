import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
x,y=make_moons(n_samples=1000,noise=0.05)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

xx=sc.fit_transform(x)
# plt.scatter(xx[:,0],xx[:,1])

from sklearn.cluster import DBSCAN
db=DBSCAN(eps=0.5)
model=db.fit(xx)
plt.scatter(xx[:,0],xx[:,1],c=model.labels_)
plt.show()