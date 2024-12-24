import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.io import arff
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
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

x_train = X.iloc[:3500,:]
x_test = X.iloc[3500:,:]
y_train = y.iloc[:3500]
y_test = y.iloc[3500:]

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
report = classification_report(y_test,y_pred,zero_division=0)
print(report)

#Using heatmap to describe Confusion Matrix
'''cm = confusion_matrix(y_test,y_pred)
np.fill_diagonal(cm,0)
plt.figure()
sns.heatmap(cm,annot=True,fmt='d',cmap='Reds',xticklabels=['1','2','3','4','5'],
            yticklabels=['1','2','3','4','5'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()'''

r_scores_acc= []
r_scores_recall = []
r_scores_precision = []

recall_scorer = make_scorer(recall_score,average='weighted')
precision_scorer = make_scorer(precision_score,average='weighted',zero_division=0)

scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=13)
X_trans = pca.fit_transform(X)

for i in range(10):
    rfc = RandomForestClassifier(n_estimators=41,n_jobs=-1,max_depth=15)
    r_scores_acc.append(cross_val_score(rfc,X_trans,y,cv=10).mean())

plt.figure()
plt.plot(range(1,11),r_scores_acc,label='Accuracy',color='r')
plt.legend(loc='best')
plt.xlabel('times')
plt.title('Random Forest')
plt.show()


#the number of the trees——41
'''num_tree = []
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(n_estimators=i + 1 , n_jobs=-1, random_state=9)
    score = cross_val_score(rfc,X,y,cv=5).mean()
    num_tree.append(score)
print(max(num_tree), (num_tree.index(max(num_tree)) * 10) + 1)
plt.plot(range(1, 201, 10), num_tree)
plt.show()

num_tree = []
for i in range(35, 45):
    rfc = RandomForestClassifier(n_estimators=i, n_jobs=-1, random_state=9)
    score = cross_val_score(rfc, X, y, cv=5).mean()
    num_tree.append(score)
print((max(num_tree)), ([*range(35, 45)][num_tree.index(max(num_tree))]))
plt.plot(range(35,45), num_tree)
plt.show()'''

#max_depth——15
'''dep_tree = []
for i in range(1, 30):
    rfc = RandomForestClassifier(n_estimators=41, n_jobs=-1, random_state=9,max_depth= i )
    score = cross_val_score(rfc, X, y, cv=5).mean()
    dep_tree.append(score)
print((max(dep_tree)), ([*range(1, 30)][dep_tree.index(max(dep_tree))]))
plt.plot(range(1,30), dep_tree)
plt.show()'''

#pca——13
'''pca_tree = []
scaler = StandardScaler()
X = scaler.fit_transform(X)
for i in range(10,140,10):
    pca = PCA(n_components=i)
    X_trans = pca.fit_transform(X)
    rfc = RandomForestClassifier(n_estimators=41,n_jobs=-1,random_state=9,max_depth=15)
    score = cross_val_score(rfc,X_trans,y,cv=5).mean()
    pca_tree.append(score)
print(max(pca_tree), (pca_tree.index(max(pca_tree)) * 10) + 1)
plt.plot(range(11,141,10), pca_tree)
plt.show()

pca_tree = []
scaler = StandardScaler()
X = scaler.fit_transform(X)
for i in range(1,20):
    pca = PCA(n_components=i)
    X_trans = pca.fit_transform(X)
    rfc = RandomForestClassifier(n_estimators=41,n_jobs=-1,random_state=9,max_depth=15)
    score = cross_val_score(rfc,X_trans,y,cv=5).mean()
    pca_tree.append(score)
print((max(pca_tree)), ([*range(1, 15)][pca_tree.index(max(pca_tree))]))
plt.plot(range(1,20), pca_tree)
plt.show()'''

