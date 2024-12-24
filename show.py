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

counter = Counter(y_test)
labels = list(counter.keys())
frequencies = list(counter.values())
plt.figure()
plt.bar(labels, frequencies, color='green')
plt.title('Frequency of class')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

'''plt.figure()
plt.plot(range(140),X.iloc[412])
plt.show()'''