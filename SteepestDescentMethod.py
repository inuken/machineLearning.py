#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      inukenta
#
# Created:     22/05/2015
# Copyright:   (c) inukenta 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#1変数の最急降下法
import random
from sympy import *

learningRate = 0.1
eps = 1e-10
iteration_max = 1000

def SteepestDescentMethod(f):
    x_init  = random.randint(-100,100)
    df = diff(f,Symbol("x"))

    for i in range(iteration_max):
        x_new = x_init - learningRate * df.subs([(x,x_init)])

        if abs(x_init - x_new) < eps:
            break

        x_init = x_new
    return x_new

if __name__ == '__main__':
    x = Symbol("x")
    f = x**2  - 4 * x + 5
    minx = SteepestDescentMethod(f)
    print (minx,f.subs([(x,minx)]))

