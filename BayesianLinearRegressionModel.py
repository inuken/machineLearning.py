'''
Created on 2015/06/26

@author: inukenta
'''
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])

def phi(x):
    s = 0.1
    return np.append(1, np.exp(-(x - np.arange(0, 1 + s, s)) ** 2 / (2 * s * s)))

PHI = np.array([phi(x) for x in X])

alpha = 0.1
beta = 9.0

Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, t))

def normal_dist_pdf(x, mean, var):
    return np.exp(-(x-mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

def quad_form(A, x):
    return np.dot(x, np.dot(A, x))

xlist = np.arange(0, 1, 0.01)
tlist = np.arange(-1.5, 1.5, 0.01)
z = np.array([normal_dist_pdf(tlist, np.dot(mu_N, phi(x)),
        1 / beta + quad_form(Sigma_N, phi(x))) for x in xlist]).T

plt.contourf(xlist, tlist, z, 5, cmap=plt.cm.binary)
plt.plot(xlist, [np.dot(mu_N, phi(x)) for x in xlist], 'r')
plt.plot(X, t, 'go')
plt.show()