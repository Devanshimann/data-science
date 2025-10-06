import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder


df=pd.read_csv("C:\\Users\\itsde\\OneDrive\\Desktop\\practice\\drug200.csv")
# print(df)
df.isna().sum()
df[df.duplicated]

sns.countplot(data=df,x='Sex',color='red')
# plt.show()
oe=OrdinalEncoder()
df['Sex']=oe.fit_transform(df[['Sex']])
df['Age']=oe.fit_transform(df[['Age']])
df['BP']=oe.fit_transform(df[['BP']])
df['Drug']=oe.fit_transform(df[['Drug']])
df['Cholesterol']=oe.fit_transform(df[['Cholesterol']])
df['Na_to_K']=oe.fit_transform(df[['Na_to_K']])
# print(df)
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
# print(x.shape)
# print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
tree1=DecisionTreeClassifier(criterion='gini')
tree1.fit(x_train,y_train)
prid=tree1.predict(x_test)
accu=accuracy_score(prid,y_test)
print(accu)
from sklearn import tree
plt.figure(figsize=(10,10))
tree.plot_tree(tree1.fit(x_train,y_train))
plt.show()
