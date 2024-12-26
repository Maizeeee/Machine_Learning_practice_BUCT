from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.io import arff
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score,silhouette_samples
from matplotlib.pyplot import nipy_spectral


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
gmm = GaussianMixture(n_components=5)
gmm.fit(X)
y_pred_train = gmm.predict(x_train) + 1
y_pred = gmm.predict(X) + 1

# #acc
# def map_label(y_train,y_pred_train,n_classes):
#     label_map = {}
#     c = Counter(y_train)
#     ct = Counter(y_pred_train)
#     c = list(c.items())
#     c.sort(key=lambda x:x[1])
#     ct = list(ct.items())
#     ct.sort(key=lambda x:x[1])
#     for i in range(n_classes):
#         label_map[ct[i][0]] =c[i][0]
#
#     return label_map
#
# label_map = map_label(y_train,y_pred_train,5)
# y_pred_test = gmm.predict(x_test) + 1
# for i in range(1500):
#     y_pred_test[i] = label_map[y_pred_test[i]]
#
# report = classification_report(y_test,y_pred_test)
# print(report)
#
# y_pred = gmm.predict(X) + 1
#
# #confusion matrix
# cm = confusion_matrix(y,y_pred)
# np.fill_diagonal(cm,0)
# plt.figure()
# sns.heatmap(cm,annot=True,fmt='d',cmap='Reds',xticklabels=['1','2','3','4','5'],
#             yticklabels=['1','2','3','4','5'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()
#
# kf = KFold(n_splits=10,shuffle=True)
# gmm = GaussianMixture(n_components=5)
#
# r_acc = []
# for i in range(1,11):
#     acc = []
#     for train_index,test_index in kf.split(X):
#         X_train,X_test = X.iloc[train_index,:],X.iloc[test_index,:]
#         y_train,y_test = y.iloc[train_index],y.iloc[test_index]
#         gmm.fit(X_train)
#         y_pred_train = gmm.predict(X_train)
#         label_map = map_label(y_train,y_pred_train,5)
#         y_pred = gmm.predict(X_test)
#         for i in range(len(y_pred)):
#             y_pred[i] = label_map[y_pred[i]]
#         acc.append(accuracy_score(y_test,y_pred))
#     r_acc.append(sum(acc)/len(acc))
#
# plt.figure()
# plt.plot(range(1,11),r_acc)
# plt.show()

#silhouette
silhouette_avg = silhouette_score(X, y_pred)
print(f"轮廓系数：{silhouette_avg}")
silhouette_vals = silhouette_samples(X, y_pred)

# image of silhouette
y_lower = 10
for i in range(1,6):
    cluster_silhouette_vals = silhouette_vals[y_pred == i]
    cluster_silhouette_vals.sort()
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = 'green'
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.yticks([])
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()




