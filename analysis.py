import pandas as pd

df = pd.read_csv("cleanned.csv")
#print(df.head())
#print(df.info())
#print(df.describe())

#meaningful subset of data
oldmen = df[(df['age']>=0) & (df['sex']=='Male')]
print(oldmen)
print(oldmen.shape)
