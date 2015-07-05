'''
Created on 2015/07/05

@author: inuken
'''

import numpy as np

X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])

equation = 3

def phi(x):
    return [1 if i == 0 else x ** i for i in range(equation)]

PHI = np.array([phi(x) for x in X])
w = np.linalg.solve(np.dot(PHI.T,PHI), np.dot(PHI.T,t))