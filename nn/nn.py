'''
Created on 2015/08/16

@author: inuken
'''

import numpy as np
import random

def sigmoid(x, beta = 1.0):
    return 1.0 / (1.0 + np.exp(beta * -x))

def d_sigmoid(x, beta = 1.0):
    return beta * x * (1 - x)

def linear(x):
    return x

def d_linear(x):
    return 1

class NeuralNetwork:
    def __init__(self, in_size, h_size, o_size):
        self.w = np.random.random([h_size, in_size + 1]) * 2 - 1
        self.v = np.random.random([o_size, h_size + 1])  * 2 - 1

    def fire(self, x):
        h = sigmoid(np.dot(self.w, np.insert(x, obj=0, values=1.0).T))
        o = linear(np.dot(self.v, np.insert(h, obj=0, values=1.0).T))
        return h, o

    def backPropagation(self, x, t, lr):
        h, o  = self.fire(x)

        delta_o = d_linear(o) * (t - o)
        delta_h = d_sigmoid(np.insert(h, obj=0, values=1.0, axis=0)) * np.dot(delta_o, self.v)

        d_v = lr * np.outer(delta_o, np.insert(h, obj=0, values=1.0, axis=0))
        self.v = self.v + d_v

        d_w = lr * np.outer(delta_h[1:], np.insert(x, obj=0, values=1.0, axis=0))
        self.w = self.w + d_w

    def learning(self, X, T, iteration):
        list = range(len(X))

        for i in range(iteration):
            random.shuffle(list)

            for num in list :
                self.backPropagation(X[num], T[num], 1.0 - (i / iteration))

    def test(self, x):
        h = sigmoid(np.dot(self.w, np.insert(x, obj = 0, values = 1.0).T))
        o = linear(np.dot(self.v, np.insert(h, obj = 0, values = 1.0).T))
        return o

if __name__ == "__main__":
    X =[[0,0],[0,1],[1,0],[1,1]]
    T = [0,5,5,2]
    nn = NeuralNetwork(2,10,1)
    nn.learning(X,T,3000)

    for x in X :
        print str(x) + "\t" + str(nn.test(x))