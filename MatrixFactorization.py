#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      inuken
#
# Created:     19/05/2015
# Copyright:   (c) inuken 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from numpy import *

topic = 4
iteration = 1000

def difcost(a,b):
    dif = 0
    #行列のすべての行と列をループする
    for i in range(shape(a)[0]) :
        for j in range(shape(a)[1]):
            #差をたし合わせる
            dif += pow(a[i,j] - b[i,j],2)
    return dif

def factorize(v):
    #行列をランダムな値で初期化
    w = matrix([[random.random() for i in range(topic)] for j in  range(shape(v)[0])])
    h = matrix([[random.random() for i in range(shape(v)[1])] for j in  range(topic)])

    #最大でiterationの回数だけ操作を繰り返す
    for i in range(iteration):
        wh = w * h

        #現在の差を計算
        cost = difcost(v,wh)

        #行列が完全に因子分解されたら終了
        if cost == 0:
            break

        h = matrix(array(h)*array(transpose(w)*v)/array(transpose(w)*w*h))
        w = matrix(array(w)*array(v*transpose(h))/array(w*h*transpose(h)))
    return w,h

if __name__=='__main__':
    mat = matrix([[5,0,6,2,2,9,0],
    [7,5,0,1,0,6,0],
    [4,3,0,0,0,5,0],
    [1,0,1,8,1,8,1],
    [2,0,4,3,6,6,0]])

    w,h = factorize(mat)

    #分解された行列を出力
    print ('w=')
    for i in range(shape(w)[0]):
        for j in range(shape(w)[1]):
            print('%3f' % w[i,j],end=","),
        print()
    print()
    print ('h=')
    for i in range(shape(h)[0]):
        for j in range(shape(h)[1]):
            print('%3f' % h[i,j],end=","),
        print()
    print()
    #元々の行列を出力
    print ('answer=')
    print (mat)
    print()
    #分解された行列2つをかけあわせた答えを出力
    print ('w*h=')
    wh = w * h
    for i in range(shape(wh)[0]):
        for j in range(shape(wh)[1]):
            print('%3f' % wh[i,j],end=","),
        print()
