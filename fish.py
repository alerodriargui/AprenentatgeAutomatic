import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
data = np.loadtxt('fish.txt')
X = data[:,1:-1]
y = data[:,-1]
# indices of classes
i0 = np.where(y == 0)[0]
i1 = np.where(y == 1)[0]
# class samples
X0 = X[i0,:]
y0 = y[i0]
X1 = X[i1,:]
y1 = y[i1]
# number of samples for each class
print('number of samples class 0: ',
X0.shape[0], y0.shape[0])
print('number of samples class 1: ',
X1.shape[0], y1.shape[0])

# ML model: straight line
clf = svm.LinearSVC(fit_intercept=True, random_state=0)
clf.fit(X, y) # training
# get the model parameters
w = clf.coef_[0]
a, b = w[0], w[1]
c = clf.intercept_[0]
print('a = %.3f, b = %.3f, c = %.3f' % (a, b, c))
# plotting
plt.figure()
# plot samples
plt.scatter(X0[:,0],X0[:,1],label='class 0')
plt.scatter(X1[:,0],X1[:,1],label='class 1')
# plot model
yy = np.linspace(X[:,1].min(),X[:,1].max(),100)
plt.plot(-b/a * yy - c/a, yy, 'k:')
plt.legend()
plt.show()
# make two predictions
p = [4, 22]
l = clf.predict([p])[0]
print('predicted label for p = {} is {}'.format(p, l))
q = [6, 15]
l = clf.predict([q])[0]
print('predicted label for q = {} is {}'.format(q, l))