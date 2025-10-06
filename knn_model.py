import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
data=datasets.load_wine(as_frame=True)
# print(data)
x=data.data
y=data.target
df=pd.DataFrame(x,columns=data.feature_names)
df["class"]=data.target
df["class"]=df["class"].replace(to_replace=[0,1,2],value=["class0","class1","class2"])

# sns.pairplot(data=df,hue="class",palette="Set2")
# # plt.show()
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=1)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_text=sc.fit_transform(x_text)
from sklearn.neighbors import KNeighborsClassifier
import math
math.sqrt(len(y_test))
knn=KNeighborsClassifier(n_neighbors=7,metric='manhattan')
knn.fit(x_train,y_train)
pred=knn.predict(x_text)
from sklearn import metrics
score=metrics.accuracy_score(y_test,pred)
print(score)

