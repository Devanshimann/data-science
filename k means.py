import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.cluster import KMeans
x,y=make_blobs(n_samples=2000,centers=4,n_features=3,random_state=23)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=23)
wcss=[]
for k in range(1,12):
    kmeans=KMeans(n_clusters=k,init="k-means++",random_state=23)
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)
print(wcss)
# plt.plot(range(1,12),wcss)
# plt.xlabel("range of k")
# plt.ylabel("wcss values")
# # plt.show()
kmeans1=KMeans(n_clusters=4,init="k-means++",random_state=23)
label=kmeans1.fit_predict(x_train)

# plt.scatter(x_train[:,0],x_train[:,1],x_train[:,2],c=label)
from kneed import KneeLocator
k=KneeLocator(range(1,12),wcss,curve="convex",direction="decreasing")
print(k.elbow)
from sklearn.metrics import silhouette_score
coff=[]
for k in range(2,12):
    kmean=KMeans(n_clusters=k,init="k-means++",random_state=23)
    kmean.fit(x_train)
    score=silhouette_score(x_train,kmean.labels_)
    coff.append(score)

print(coff)
predict=kmeans1.predict(x_test)
plt.scatter(x_test[:,0],x_test[:,1],x_test[:,2],c=predict)
plt.show()
