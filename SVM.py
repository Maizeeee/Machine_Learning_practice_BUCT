import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.io import arff
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier


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


#classifier
classifier = OneVsRestClassifier(SVC())
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
report = classification_report(y_test,y_pred,zero_division=0)
print(report)

#Using heatmap to describe Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
np.fill_diagonal(cm,0)
plt.figure()
sns.heatmap(cm,annot=True,fmt='d',cmap='Reds',xticklabels=['1','2','3','4','5'],
            yticklabels=['1','2','3','4','5'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# svc = SVC()
# model = OneVsRestClassifier(estimator=svc)
#
# r_scores_acc = []

# for i in range(10):
#     classifier = OneVsRestClassifier(SVC(kernel='rbf',C=1,gamma=0.01))
#     r_scores_acc.append(cross_val_score(classifier,X,y,cv=5).mean())
#
# plt.figure()
# plt.plot(range(1,11),r_scores_acc,label='Accuracy',color='r')
# plt.xlabel('times')
# plt.title('SVM')
# plt.show()

#search the best parameter
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['linear', 'rbf']
# }
# for C in param_grid ['C']:
#     for gamma in param_grid['gamma']:
#         for kernel in param_grid['kernel']:
#             svc.set_params(C=C,gamma=gamma,kernel=kernel)
#             model.fit(x_train,y_train)
#             y_pred = model.predict(x_test)
#             acc = accuracy_score(y_test,y_pred)
#             r_scores_acc.append((kernel,C,gamma,acc))
#
# r_scores_acc = np.array(r_scores_acc)
# plt.figure()
# for kernel in np.unique(r_scores_acc[:,0]):
#     kernel_scores = r_scores_acc[r_scores_acc[:,0]==kernel]
#     plt.plot(kernel_scores[:,2],kernel_scores[:,3],label=f'{kernel}kernel')
#
# plt.legend(loc='best')
# plt.show()