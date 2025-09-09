import pandas as pd

df = pd.read_csv("cleanned.csv")

'''
print(df.head())
print(df.info())
print(df.describe())

want to see data specifically for men 60 years or older
oldmen = df[(df['age']>=60) & (df['sex']=='Male')]
print(oldmen)
print(oldmen.shape)
'''
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