import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
df=pd.read_csv("C:\\Users\\itsde\\OneDrive\\Desktop\\practice\\tested.csv")

# print(df.columns)

df.drop_duplicates()


# df['Age'].fillna(df['Age'].mean(),inplace=True)
# df['Fare'].fillna(df['Fare'].mean(),inplace=True)
# print(df['Cabin'])
threshold=60
per=df.isnull().sum()/len(df)*100
todrop=per[per>threshold].index
# # print(todrop)
# df.drop(columns=todrop,inplace=True)
# print(df.head())
# print(df.isnull().sum())
# print(df['Embarked'].value_counts())
# print(df.describe())
# sea.boxplot(x=df['Fare'])
# plt.title("fare distribution")
# sea.barplot(x="Pclass",y="Fare",data=df,palette="coolwarm")
# plt.title("Average fare by class")
# sea.histplot(data=df,x=df['Age'],hue=df["Survived"],color="#BF8ED2",bins=5)
# sea.boxplot(x="Survived", y="Age", data=df, palette="Set2")
# plt.title("SURVIVAL BY AGE")
# plt.xlabel("Survived or not")             
# plt.ylabel("Survived by age")

# plt.show()
# sea.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette="coolwarm")
# plt.title("Age vs Fare by Survival")
# plt.show()

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'] = df['Age'].astype(int)

nal=df.groupby(['Age',"Pclass"])["Fare"].mean()
print(nal)


