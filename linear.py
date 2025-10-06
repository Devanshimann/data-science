import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
data = datasets.load_diabetes(as_frame=True)

x=data.data
y=data.target
df=pd.DataFrame(x,columns=data.feature_names)
df['target']=y
from sklearn import linear_model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
x_test.shape
lin=linear_model.LinearRegression()
lin.fit(x_train,y_train)
pred=lin.predict(x_test)
print(pred)
plt.scatter(pred,y_test)
print("accuracy score is ",lin.score(x,y))

mse = metrics.mean_squared_error(y_test, pred)
print(mse)