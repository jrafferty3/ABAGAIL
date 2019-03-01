import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import preprocessing 
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


splits = range(1, 501)

# MLP
mlp_adult_train_scores = []
mlp_adult_test_scores = []

adult = pd.read_csv('./adult.data',header=None)
adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']
print(adult.ix[adult.cap_gain>0].cap_loss.abs().max())
print(adult.ix[adult.cap_loss>0].cap_gain.abs().max())
adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
adult = adult.drop(['employer', 'fnlwt','edu','marital','occupation','relationship','race','sex','cap_gain','cap_loss','country'],1)
adult['income'] = pd.get_dummies(adult.income) 
adult = adult[['age', 'edu_num','hrs','cap_gain_loss','income']]


a_Y = adult.income
x = adult.drop(['income'], axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
a_X_scaled = pd.DataFrame(min_max_scaler.fit_transform(x))

start = time.time()

a_training_set_sizes = []
d_training_set_sizes = []

ax_train, ax_test, ay_train, ay_test = train_test_split(a_X_scaled, a_Y, test_size=0.7, random_state=43)
    
a_training_set_sizes.append(len(ax_train))

for split in splits:
    # MLP
    a_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = [14, 14, 14, 14], random_state=1, max_iter=split)
    a_mlp.fit(ax_train, ay_train)
    
    ay_pred = [k[1] for k in a_mlp.predict_proba(ax_train)]
    mse = mean_squared_error(ay_train, ay_pred)
    mlp_adult_train_scores.append(mse)
    
    ay_pred = [k[1] for k in a_mlp.predict_proba(ax_test)]
    mse = mean_squared_error(ay_test, ay_pred)
    mlp_adult_test_scores.append(mse)
    
    

with open("timings.txt", "a") as myfile:
    myfile.write("Analysis - {}\n".format(time.time() - start))


results_df = pd.DataFrame()
results_df["mlp_adult_test_scores"] = mlp_adult_test_scores
results_df["mlp_adult_train_scores"] = mlp_adult_train_scores


results_df.to_csv("MLP_results.csv")