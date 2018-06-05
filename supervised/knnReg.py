import mglearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y= mglearn.datasets.make_wave(n_samples= 40)
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state= 0)
reg= KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print(reg.predict(X_test))  
print(reg.score(X_test, y_test))

import os  
dmode = os.environ.get('DISPLAY', '')  
  
if dmode:  
    import matplotlib.pyplot as plt  
    import numpy as np  
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  
    # create 1000 data points, evenly spaced between -3 and 3  
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)  
    plt.suptitle("nearest_neighbor_regression")  
    for n_neighbors, ax in zip([1, 3, 9], axes):  
        # make predictions using 1, 3 or 9 neighbors  
        reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)  
        ax.plot(X, y, 'o')  
        ax.plot(X, -3 * np.ones(len(X)), 'o')  
        ax.plot(line, reg.predict(line))  
        ax.set_title("%d neighbor(s)" % n_neighbors)  
  
    plt.show()  

mglearn.plots.plot_knn_regression(n_neighbors=1)

