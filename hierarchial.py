import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
iris=datasets.load_iris()
data=pd.DataFrame(iris.data)
data.columns=iris.feature_names
y=iris.target
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2,random_state=23)
st=StandardScaler()
x=st.fit_transform(x_train)
xtest=st.fit_transform(x_test)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
xx=pca.fit_transform(x)
x2=pca.fit_transform(xtest)
# plt.scatter(xx[:,0],xx[:,1])
import scipy.cluster.hierarchy as sc
# sc.dendrogram(sc.linkage(xx,method="ward"))
# plt.title(":dendogram")
from sklearn.cluster import AgglomerativeClustering
aglo=AgglomerativeClustering(n_clusters=2,linkage="ward")
aglo.fit(xx)
print(aglo.labels_)
# plt.scatter(xx[:,0],xx[:,1],c=aglo.labels_)
from sklearn.metrics import silhouette_score
coff=[]
for k in range(2,11):
    agloo=AgglomerativeClustering(n_clusters=k,linkage="ward")
    agloo.fit(xx)
    score=silhouette_score(xx,agloo.labels_)
    coff.append(score)
plt.plot(range(2,11),coff)
plt.title("silhouette scores")
plt.show()