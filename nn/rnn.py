#-*-coding:utf-8-*-
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

class RecurrentNeuralNetwork:
    def __init__(self, in_size, h_size, o_size):
        self.u = np.random.random([h_size, in_size + 1]) * 2 - 1
        self.v = np.random.random([o_size, h_size + 1]) * 2 - 1
        self.w = np.random.random([h_size, h_size + 1]) * 2 - 1
        self.hOld = [0] * h_size

    def fire(self, x):
        h = sigmoid(np.dot(self.u, np.insert(x, obj=0, values=1.0).T) + np.dot(self.w, np.insert(self.hOld, obj=0, values=1.0).T))
        o = linear(np.dot(self.v, np.insert(h, obj=0, values=1.0).T))
        return h, o

    def backPropagationThroughTime(self, x, t, lr):
        h, o  = self.fire(x)

        delta_o = d_linear(o) * (t - o)
        delta_h = d_sigmoid(np.insert(h, obj=0, values=1.0, axis=0)) * np.dot(delta_o, self.v)

        d_v = lr * np.outer(delta_o, np.insert(h, obj=0, values=1.0, axis=0))
        self.v = self.v + d_v

        d_u = lr * np.outer(delta_h[1:], np.insert(x, obj=0, values=1.0, axis=0))
        self.u = self.u + d_u

        d_w = lr * np.outer(delta_h[1:], np.insert(self.hOld, obj=0, values=1.0, axis=0))
        self.w = self.w + d_w

        self.hOld = h

    def learning(self, X, T, iteration):
        list = range(len(X))

        for i in range(iteration):
            random.shuffle(list)

            for num in list :
                self.backPropagationThroughTime(X[num], T[num], 1.0 - (i / iteration))

    def test(self, x):
        h = sigmoid(np.dot(self.u, np.insert(x, obj=0, values=1.0).T) + np.dot(self.w, np.insert(self.hOld, obj=0, values=1.0).T))
        o = linear(np.dot(self.v, np.insert(h, obj=0, values=1.0).T))
        return o

if __name__ == "__main__":
    X = [[0,0],[0,1],[1,0],[1,1]]
    T = [[0,1],[1,0],[1,1],[0,0]]
    rnn = RecurrentNeuralNetwork(2,5,2)
    rnn.learning(X,T,3000)
    for x in X :
        print str(x) + "\t" + str(rnn.test(x))