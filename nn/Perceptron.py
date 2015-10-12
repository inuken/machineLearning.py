'''
Created on 2015/06/29

@author: inuken
'''

import numpy as np
import matplotlib.pyplot as plt
import random

N = 100

np.random.seed(0)

X = np.random.randn(N, 2)

def h(x, y):
    return 5 * x + 3 * y - 1

T = np.array([1 if h(x, y) > 0 else -1 for x, y in X])

def phi(x, y):
    return np.array([x, y, 1])

w= np.zeros(3)

while True:
    lis = range(N)
    random.shuffle(lis)

    misses = 0
    for n in lis :
        x_n, y_n = X[n, :]
        t_n = T[n]

        predict  = np.sign((w * phi(x_n, y_n)).sum())

        if predict != t_n:
            w += t_n * phi(x_n, y_n)
            misses += 1

    if misses == 0:
        break

seq = np.arange(-3, 3, 0.02)
xlist, ylist = np.meshgrid(seq, seq)
zlist = np.sign((w * phi(xlist, ylist)).sum())

plt.pcolor(xlist, ylist, zlist, alpha=0.2, edgecolors='white')
plt.plot(X[T== 1,0], X[T== 1,1], 'o', color='red')
plt.plot(X[T==-1,0], X[T==-1,1], 'o', color='blue')
plt.show()
