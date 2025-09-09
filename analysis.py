import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("cleanned.csv")
'''
print(df.head())
print(df.info())
print(df.describe())

#want to see data specifically for men 60 years or older
oldmen = df[(df['age']>=60) & (df['sex']=='Male')]
print(oldmen)
print(oldmen.shape)


#testing out grouping
sexChol = df.groupby('sex')['chol'].agg(
    q1 = lambda x: x.quantile(.25),
    q3 = lambda x: x.quantile(.75),
    mean = 'mean',
    med ='median',
    min = 'min',
    max = 'max', 
    cnt = 'count'
    )
print(sexChol)

'''
#attempt at random forest 
#first I set my dependent/outcome (y) and independent/predictor variables(x)
x = df.drop("num", axis=1) 
y = df["num"]

x = pd.get_dummies(x, drop_first=True) #update x to set dummy variables for discrete variables


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=24
)

#I made 50 decision trees and used the same random seed as before so I can reproduce my results
rf = RandomForestClassifier(n_estimators=50, random_state=24)

#train 80% of the x data on 80% of the y data
rf.fit(x_train, y_train)

#get the random forest prediction of the y_test given the x_test data
y_prediction = rf.predict(x_test)

print("accuracy is " + str(accuracy_score(y_test, y_prediction)))





