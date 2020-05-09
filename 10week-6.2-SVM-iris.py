# -*- coding: utf-8 -*-
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
#%matplotlib inline
iris = datasets.load_iris()
print(iris.keys())
print(iris.data.shape)
print(iris.feature_names)
print(iris.DESCR)
x = iris.data[:, :2]
y = iris.target
SVM = svm.SVC(kernel='linear', C=1).fit(x, y) # linear kernel
#SVM = svm.SVC(kernel='rbf', C=1).fit(x, y) # rbf
#SVM = svm.SVC(kernel='rbf', C=1, gamma=10).fit(x, y) # rbf, gamma=10
#SVM = svm.SVC(kernel='rbf', C=1, gamma=100).fit(x, y) # rbf, gamma=100
#SVM = svm.SVC(kernel='rbf', C=1, gamma="auto").fit(x, y) # rbf, gamma=auto
#SVM = svm.SVC(kernel='rbf', C=100, gamma="auto").fit(x, y) # rbf
#SVM = svm.SVC(kernel='rbf', C=1000).fit(x, y) # rbf
x_min, x_max = x[:, 0].min()-1, x[:, 0].max()+1
y_min, y_max = x[:, 1].min()-1, x[:, 1].max()+1
plot_unit = 0.025
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_unit), np.arange(y_min, y_max, plot_unit))
temp=np.c_[xx.ravel(), yy.ravel()]
z = SVM.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
print('정확도 : ',SVM.score(X = x, y = y))
# =============================================================================
# plotting
# =============================================================================
plt.pcolormesh(xx, yy, z, alpha=0.1)
plt.scatter(x[:, 0],x[:, 1],c=y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Machine')
plt.show()


