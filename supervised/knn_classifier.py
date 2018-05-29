#loading dataset
from sklearn.datasets import load_iris

iris= load_iris()
print(iris.keys())
print(iris['target_names'])
print(iris['feature_names'])
print(type(iris['data']))
print(iris['data'])
print(iris['data'].shape)
print(type(iris['target']))
print(iris['target'])
print(iris['target'].shape)

#splitting dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(iris['data'], iris['target'], random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#graphical representation of the dataset
import matplotlib.pyplot as plt

fig, ax= plt.subplots(3, 3, figsize= (15, 15))
plt.suptitle("iris_pairplot")

for i in range(3):
   for j in range(3):
      ax[i,j].scatter(X_train[:, j], X_train[:, i+1], c=y_train, s=60)
      ax[i,j].set_xticks(())
      ax[i,j].set_yticks(())
      if i==2:
          ax[i,j].set_xlabel(iris['feature_names'][j])
      if j==0:
          ax[i,j].set_ylabel(iris['feature_names'][i+1])
      if j>i:
          ax[i,j].set_visible(False)

plt.show()


#applying KNN classifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

knn= KNeighborsClassifier(n_neighbors=1)#instantiate class
knn.fit(X_train, y_train)#fit the classifier
KNeighborsClassifier(algorithm= 'auto', leaf_size= 30, metric='minkowski', metric_params= None, n_jobs= 1, n_neighbors= 1, p= 2, weights= 'uniform')

X_new= np.array([[5, 2.9, 1, 0.2]])#make prediction 
print(X_new.shape)
pred= knn.predict(X_new)
print(pred)
print(iris['target_names'][pred])


y_pred= knn.predict(X_test)#make prediction on test dataset
print(y_pred)
print(iris['target_names'][y_pred])


print(knn.score(X_test, y_test))#score
