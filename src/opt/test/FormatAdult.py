import pandas as pd
import numpy as np
from sklearn import preprocessing 

### The following importing and processing of the adult dataset was taken from 
### https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/parse%20data.py
# Preprocess with adult dataset
adult = pd.read_csv('./adult.data',header=None)
adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']
print(adult.ix[adult.cap_gain>0].cap_loss.abs().max())
print(adult.ix[adult.cap_loss>0].cap_gain.abs().max())
adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
adult = adult.drop(['employer', 'fnlwt','edu','marital','occupation','relationship','race','sex','cap_gain','cap_loss','country'],1)
adult['income'] = pd.get_dummies(adult.income) 
adult = adult[['age', 'edu_num','hrs','cap_gain_loss','income']]


x  = adult.values
min_max_scaler = preprocessing.MinMaxScaler()
adult = pd.DataFrame(min_max_scaler.fit_transform(x))

#adult = adult.rename(columns=lambda x: x.replace('-','_'))
adult.to_csv("adult.csv", index=False, header=False)