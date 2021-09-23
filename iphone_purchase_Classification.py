# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:35:29 2021

@author: doguilmak

dataset: https://github.com/omairaasim/machine_learning/tree/master/project_14_naive_bayes

"""
#%%
# 1. Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")

#%%
# 2. Data Preprocessing

# 2.1. Importing Data
start = time.time()
datas = pd.read_csv('iphone_purchase_records.csv')

# 2.2. Looking For Anomalies
print(datas.info()) # Looking for the missing values
print(datas.describe().T)
print(datas.isnull().sum())

# 2.3. Creating correlation matrix heat map and examine relationship between datas
"""
Plot rectangular data as a color-encoded matrix.
https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""
plt.figure(figsize = (12, 6))
sns.heatmap(datas.corr(),annot = True, cmap="Greens")
sns.pairplot(datas, hue = 'Purchase Iphone', aspect = 1.5)
plt.show()

# 3.4. Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
datas["Gender"] = le.fit_transform(datas['Gender'])

# 2.5. DataFrame Slice - Determination of Dependent and Independent Variables
y = datas.iloc[:, 3:4]
x = datas.iloc[:, 0:3]  

# 2.6. NumPy Array Translate
X = x.values
Y = y.values

# 2.7. Train - Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0) 

# 2.8. Scaling Datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%%
# Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('\nLogistic Regression Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(logr, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Logistic Regression Classifier')  
plt.show()

# Predicting
predict = np.array([1, 19, 19000]).reshape(1, 3)
y_pred = logr.predict(predict)
if y_pred == 0:
    print("The model predicted as he/she didn't get Iphone. Class: ", y_pred)
else:
    print('The model predicted as he/she get Iphone. Class: ', y_pred)

#%%
# K-NN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix - Accuracy Score - K-NN Score
print('\nK-NN Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"K-NN Score: {knn.score(X_test, y_test)}")

# Plotting Confusion Matrix
plot_confusion_matrix(knn, X_test, y_test)
plt.title('K-NN Classifier')  
plt.show()

# Predicting
predict = np.array([1, 19, 19000]).reshape(1, 3)
y_pred = knn.predict(predict)
if y_pred == 0:
    print("The model predicted as he/she didn't get Iphone. Class: ", y_pred)
else:
    print('The model predicted as he/she get Iphone. Class: ', y_pred)

#%%
# SVM

from sklearn.svm import SVC
svc = SVC(kernel='sigmoid', C=4, gamma=0.1) # linear-rbf-sigmoid-precomputed-callable
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('\nSVC Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(svc, X_test, y_test, cmap=plt.cm.Blues)
plt.title('SVC Classifier')  
plt.show()

# Predicting
predict = np.array([1, 19, 19000]).reshape(1, 3)
y_pred = svc.predict(predict)
if y_pred == 0:
    print("The model predicted as he/she didn't get Iphone. Class: ", y_pred)
else:
    print('The model predicted as he/she get Iphone. Class: ', y_pred)
    
#%%
# Desicion Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test) 

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('\nDecision Tree Classifier Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
tree.plot_tree(dtc) 

# Plotting Confusion Matrix
plot_confusion_matrix(dtc, X_test, y_test)
plt.title('Decision Tree Classifier')
plt.show()

# Predicting
predict = np.array([1, 19, 19000]).reshape(1, 3)
y_pred = dtc.predict(predict)
if y_pred == 0:
    print("The model predicted as he/she didn't get Iphone. Class: ", y_pred)
else:
    print('The model predicted as he/she get Iphone. Class: ', y_pred)

#%%
# Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train) 

y_pred = rfc.predict(X_test)

# Confusion Matrix - Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('\nRandom Forest Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(dtc, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Random Forest Classifier')
plt.show()

# Predicting
predict = np.array([1, 19, 19000]).reshape(1, 3)
y_pred = rfc.predict(predict)
if y_pred == 0:
    print("The model predicted as he/she didn't get Iphone. Class: ", y_pred)
else:
    print('The model predicted as he/she get Iphone. Class: ', y_pred)

#%%
# K-NN Visualization

from sklearn import neighbors
from mlxtend.plotting import plot_decision_regions

def knn_comparison(data, k, m="euclidean"):
  x = data[['Age', 'Salary']].values
  y = data['Purchase Iphone'].astype('category').cat.codes.to_numpy()
  clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric = m)
  clf.fit(x, y)
  plot_decision_regions(x, y, clf=clf, legend=2)
  plt.title('K-NN with K='+ str(k))
  plt.show()

for i in [1, 5, 10, 20, 50]:
    knn_comparison(datas, i)
    
for i in [1, 5, 10, 20, 50]:
    knn_comparison(datas, i, "manhattan")
 
#%%
# ROC , TPR, FPR Values

from sklearn import metrics
print("\nPredict Probability")
y_proba = logr.predict_proba(X_test) 
print("Predict probability:\n", y_proba) 

y_pred_proba = logr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

print('AUC Score:\n', auc)
print("False Positive Rate:\n", fpr)
print("True Positive Rate:\n", tpr)

font1 = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 10,
        }
font2 = {'family': 'serif',
         'color': 'black',
         'weight': 'normal',
         'size': 15,
         }

lw = 1

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot(fpr, tpr, color='red', linestyle='-', marker='o', markerfacecolor='black', markersize=5)
plt.title("ROC", fontdict=font2)
plt.xlabel("False Positive Rate", fontdict=font1)
plt.ylabel("True Positive Rate", fontdict=font1)
plt.show()

#%%
# K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = svc, X=X_train, y=y_train, cv = 4)
print("\nK-Fold Cross Validation:")
print("Success Mean:\n", success.mean())
print("Success Standard Deviation:\n", success.std())

#%%
# Grid Search

from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5],'kernel':['linear'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5] ,'kernel':['rbf'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5] ,'kernel':['sigmoid'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5] ,'kernel':['callable'], 'gamma':[1,0.5,0.1,0.01,0.001]}]

gs = GridSearchCV(estimator=svc,
                  param_grid=p,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
best_result = grid_search.best_score_
best_parameters = grid_search.best_params_
print("\nGrid Search")
print("Best result:\n", best_result)
print("Best parameters:\n", best_parameters)
   
#%%
# Saving Model
"""
# Pickle
import pickle
file = "logr.save"
pickle.dump(logr, open(file, 'wb'))

downloaded_data = pickle.load(open(file, 'rb'))
print(downloaded_data.predict(X_test))
"""

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
