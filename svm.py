import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
iris=datasets.load_iris()
x=iris.data

y=iris.target
df=pd.DataFrame(x,columns=iris.feature_names)
df['type']=y
df['type']=df['type'].replace(to_replace=[0,1,2],value=['setosa', 'versicolor', 'virginica'])

nan=df.isna().sum()
sns.pairplot(data=df,hue='type')
# plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

svm=SVC(kernel='rbf',random_state=0)
svm.fit(x_train,y_train)
pred=svm.predict(x_test)

acc=metrics.accuracy_score(y_test,pred)
confuse=metrics.confusion_matrix(y_test,pred)
print(acc,confuse)