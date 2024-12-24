import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from scipy.io import arff
from sklearn.metrics import accuracy_score
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

'''Srandom_seed = 23
X = X.sample(frac=1,random_state=random_seed).reset_index(drop=True)
y = y.sample(frac=1,random_state=random_seed).reset_index(drop=True)

x_train = X.iloc[:3500,:]
x_test = X.iloc[3500:,:]
y_train = y.iloc[:3500]
y_test = y.iloc[3500:]'''

mlp = MLPClassifier(hidden_layer_sizes=(128,),activation='relu',solver='adam',
                    max_iter=1000)
mlp.fit(x_train,y_train)
y_pred = mlp.predict(x_test)

report = classification_report(y_test,y_pred,zero_division=0)
print(report)

cm = confusion_matrix(y_test,y_pred)
np.fill_diagonal(cm,0)
plt.figure()
sns.heatmap(cm,annot=True,fmt='d',cmap='Reds',xticklabels=['1','2','3','4','5'],
            yticklabels=['1','2','3','4','5'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

r_scores_acc= []

for i in range(10):
    mlp = MLPClassifier(hidden_layer_sizes=(128,),activation='relu',solver='adam',
                    max_iter=1000)
    r_scores_acc.append(cross_val_score(mlp,X,y,cv=10).mean())

plt.figure()
plt.plot(range(1,11),r_scores_acc,label='Accuracy',color='r')
plt.legend(loc='best')
plt.xlabel('times')
plt.title('Multilayer Perceptron')
plt.show()