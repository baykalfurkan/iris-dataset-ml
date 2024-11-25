# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:09:31 2024

@author: Kruger
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('iris.xls')

#son kolon bagımlı değisken olmalı
x = veriler.iloc[:,:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
y = y.ravel()



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#Buradan itibaren sınıflandırma algoritmaları başlar

# 1 Logistic Regresyon
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #eğitim

y_pred = logr.predict(X_test) #tahmin


#Karmaşıklık matrisi
cm = confusion_matrix(y_test,y_pred)
print(30*"-")
print("Karmasiklik matrisi")
print(cm)


# 2 KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train,y_train) #eğitim

y_pred = knn.predict(X_test) #tahmin

cm = confusion_matrix(y_test,y_pred)
print(30*"-")
print("KNN")
print(cm)


# 3 SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='rbf') 
svc.fit(X_train,y_train) #eğitim

y_pred = svc.predict(X_test) #tahmin

cm = confusion_matrix(y_test,y_pred)
print(30*"-")
print('SVC')
print(cm)


# 4 Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train) #eğitim

y_pred = gnb.predict(X_test) #tahmin

cm = confusion_matrix(y_test,y_pred)
print(30*"-")
print('GNB')
print(cm)


# 5 Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train) #eğitim
y_pred = dtc.predict(X_test) #tahmin

cm = confusion_matrix(y_test,y_pred)
print(30*"-")
print('DTC')
print(cm)


# 6 Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train) #eğitim

y_pred = rfc.predict(X_test) #tahmin
cm = confusion_matrix(y_test,y_pred)
print(30*"-")
print('RFC')
print(cm)

    
# 7 ROC , TPR, FPR değerleri 
y_proba = rfc.predict_proba(X_test) #tahmin
print(35*"-")
print(y_test)
print(y_proba[:,0])


from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(40*"-")
print(fpr)
print(tpr)



#---------------------------------------------------------------------
import matplotlib.pyplot as plt


# ROC eğrisi ve AUC değerini hesapla
roc_auc = metrics.auc(fpr, tpr)

# Grafiği çizdir
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('ROC Eğrisi: Model Performansı Karşılaştırması', fontsize=16)
plt.figure(figsize=(10, 8))
plt.show()
