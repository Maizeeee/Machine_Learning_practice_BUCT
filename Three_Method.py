import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from scipy.io import arff
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
X = scaler.fit_transform(X)

r_scores_acc_RF = []
r_scores_acc_SVM = []
r_scores_acc_MLP = []
for i in range(10):
    rfc = RandomForestClassifier()
    mlp = MLPClassifier()
    classifier = OneVsRestClassifier(SVC())
    r_scores_acc_RF.append(cross_val_score(rfc, X, y, cv=5).mean())
    r_scores_acc_MLP.append(cross_val_score(mlp,X,y,cv=5).mean())
    r_scores_acc_SVM.append(cross_val_score(classifier,X,y,cv=5).mean())

plt.figure()
plt.plot(range(1,11),r_scores_acc_RF,label='Random Forest',color='r')
plt.plot(range(1,11),r_scores_acc_SVM,label='SVM',color='b')
plt.plot(range(1,11),r_scores_acc_MLP,label='MLP',color='g')
plt.legend(loc='best')
plt.xlabel('times')
plt.title('Three Method')
plt.show()