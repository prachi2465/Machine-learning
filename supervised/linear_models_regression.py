
'''Linear regression'''

from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.model_selection import train_test_split

X, y= mglearn.datasets.make_wave(n_samples= 60)
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state= 42)
lr= LinearRegression().fit(X_train, y_train)
print("lr.coef: {}".format(lr.coef_))
print("lr.intercept: {}".format(lr.intercept_))

print("Training set score (linear regression): {:.2f} ".format(lr.score(X_train, y_train)))
print("Test set score (linear regression): {:.2f}".format(lr.score(X_test, y_test)))

#on Boston housing dataset

X, y= mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state= 0)
lr= LinearRegression().fit(X_train, y_train)
print("Training set score (linear regression): {:.2f} ".format(lr.score(X_train, y_train)))
print("Test set score (linear regression): {:.2f}".format(lr.score(X_test, y_test)))


'''Ridge regression'''

from sklearn.linear_model import Ridge
ridge= Ridge().fit(X_train, y_train)

print("Training set score (Ridge): {:.2f} ".format(ridge.score(X_train, y_train)))
print("Test set score (Ridge): {:.2f}".format(ridge.score(X_test, y_test)))

#on changing the value of alpha

ridge1= Ridge(alpha= 10).fit(X_train, y_train)
print("Training set score (Ridge): {:.2f} ".format(ridge1.score(X_train, y_train)))
print("Test set score (Ridge): {:.2f}".format(ridge1.score(X_test, y_test)))

ridge2= Ridge(alpha= 0.1).fit(X_train, y_train)
print("Training set score (Ridge): {:.2f} ".format(ridge2.score(X_train, y_train)))
print("Test set score (Ridge): {:.2f}".format(ridge2.score(X_test, y_test)))


'''Lasso'''

from sklearn.linear_model import Lasso
import numpy as np
lasso= Lasso().fit(X_train, y_train)

print("Training set score (lasso): {:.2f} ".format(lasso.score(X_train, y_train)))
print("Test set score (lasso): {:.2f}".format(lasso.score(X_test, y_test)))
print("No. of features used: {}".format(np.sum(lasso.coef_!=0)))

#on changing the value of alpha

lasso1= Lasso(alpha= 0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score (lasso): {:.2f} ".format(lasso1.score(X_train, y_train)))
print("Test set score (lasso): {:.2f}".format(lasso1.score(X_test, y_test)))
print("No. of features used: {}".format(np.sum(lasso1.coef_!=0)))

lasso2= Lasso(alpha= 0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score (lasso): {:.2f} ".format(lasso2.score(X_train, y_train)))
print("Test set score (lasso): {:.2f}".format(lasso2.score(X_test, y_test)))
print("No. of features used: {}".format(np.sum(lasso2.coef_!=0)))

