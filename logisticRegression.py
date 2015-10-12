'''
Created on 2015/06/29

@author: inuken
'''

import numpy as np
import matplotlib.pyplot as plt


N = 100

np.random.seed(0)

X = np.random.randn(N, 2)

def h(x, y):
    return 5 * x + 3 * y - 1

T = np.array([1 if h(x, y) > 0 else 0 for x, y in X])

def phi(x, y):
    return np.array([x, y, 1])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

w= np.zeros(3)

eta = 0.1

for i in xrange(50) :
    list = range(N)
    np.random.shuffle(list)

    for n in list :
        x_n, y_n = X[n, :]
        t_n = T[n]
        feature = phi(x_n, y_n)
        predict = sigmoid(np.inner(w, feature))
        w -= eta * (predict - t_n) * feature

    eta *= 0.9

seq = np.arange(-3, 3, 0.01)
xlist, ylist = np.meshgrid(seq, seq)
zlist = [sigmoid(np.inner(w,phi(x,y))) for x,y in zip(xlist,ylist)]


plt.imshow(zlist, extent = [-3,3,-3,3], origin='lower' , cmap = plt.cm.PiYG_r)
plt.plot(X[T== 1,0], X[T== 1,1], 'o', color='red')
plt.plot(X[T== 0,0], X[T== 0,1], 'o', color='blue')
plt.show()
