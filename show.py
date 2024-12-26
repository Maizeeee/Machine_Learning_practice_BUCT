from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.io import arff
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


#Dataset preprocessing
train,metar = arff.loadarff("ECG5000_TRAIN.arff")
df_train = pd.DataFrame(train)
df_train['target'] = df_train['target'].astype(int)
test,metae = arff.loadarff("ECG5000_TEST.arff")
df_test = pd.DataFrame(test)
df_test['target'] = df_test['target'].astype(int)

y_train = df_train.iloc[:,-1]
x_train = df_train.iloc[:,:-1]
y_test = df_test.iloc[:,-1]
x_test = df_test.iloc[:,:-1]

X = pd.concat([x_train,x_test],axis=0)
y = pd.concat([y_train,y_test],axis=0)

random_seed = 23
X = X.sample(frac=1,random_state=random_seed).reset_index(drop=True)
y = y.sample(frac=1,random_state=random_seed).reset_index(drop=True)

x_train = X.iloc[:3500,:]
x_test = X.iloc[3500:,:]
y_train = y.iloc[:3500]
y_test = y.iloc[3500:]

#the num of each class
counter = Counter(y_train)
labels = list(counter.keys())
frequencies = list(counter.values())
plt.figure()
plt.bar(labels, frequencies, color='green')
plt.title('Frequency of class')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

#show the shape of data
plt.figure()
plt.plot(range(140),X.iloc[412])
plt.show()