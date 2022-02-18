# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:34:30 2017

@author: alunos
"""

import numpy as np
import matplotlib.pylab as plt


a = 0.0
b = 5.0
N = 5

#parametros
dt = 1.0

h = (b-a)/(N)

#Dominio
x = np.arange(a,b+h,h)
y = np.arange(a,b+h,h)

#construindo a malha
X,Y = np.meshgrid(x,y)

numElemX = len(x)
numElemY = len(y)

#vetor auxiliar
u = np.zeros((numElemX,numElemY))
v = np.zeros((numElemX,numElemY))

U = np.zeros((numElemX,numElemY,2))
U1 = np.zeros((numElemX,numElemY,2))

#matriz auxiliar
F =  np.zeros((numElemX,numElemY,2))
for i in range(1,numElemX-1):
    for j in range(1,numElemY-1):
        F[i,j] = [1.0,1.0]
        
F = 0.5 * F
###################################
#Prescrevendo a Fronteira

#Partes Superior e Inferior do Dominios 
#sera constante e igual a zero.

#Laterais esquerda e direita 
for i in range(1,numElemX-1):
    U[i,0] = [1.0,.0]
    U[i,numElemY-1] = [1.0,.0]

U1 = U

nx = numElemX-1
ny = numElemY-1

for k in range(3):
    U1[1:nx,1:ny,:] = U[1:nx,1:ny,:] + F[1:nx,1:ny,:]
    print "*****", k , "****"
    print U1
    print "\n"
    

    u = U[:,:,0]
    v = U[:,:,1]
    plt.figure()
    for i in range(numElemX):
        for j in range(numElemY):
            plt.scatter(X[i,j],Y[i,j],c='r',s=70)
    Q = plt.quiver(X, Y, u, v)
    
    U = U1





plt.show()    