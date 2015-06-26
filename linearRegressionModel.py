'''
Created on 2015/06/25

@author: inukenta
'''
import numpy as np
import matplotlib.pyplot as plt


X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.89, -0.79, -0.04])
d = 7
r = 0.001

def phi(x):
    return [1 if i == 0 else x ** i for i in range(d)]

PHI = np.array([phi(x) for x in X])
w = np.dot(np.linalg.inv(r * np.identity(d) + np.dot(PHI.T, PHI)), np.dot(PHI.T, t))

def f(w, x):
    return np.dot(w, phi(x))

xlist = np.arange(0, 1, 0.01)
ylist = [f(w, x) for x in xlist]

plt.plot(xlist, ylist)
plt.plot(X, t, 'o')

plt.show()
