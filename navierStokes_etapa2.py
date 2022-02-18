# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:34:30 2017

@author: alunos
"""

import numpy as np
import matplotlib.pylab as plt

#########################################
#########     Funcoes ###################
def laplace(u,i,j,dx,dy):
    a = (u[i-1,j] - 2.0*u[i,j] + u[i+1,j])/(dx)**2
    b = (u[i,j-1] - 2.0*u[i,j] + u[i,j+1])/(dy)**2
    
    return a+b
    
def L(U,dx,dy):
    n,m,d = np.shape(U)    
    
    LU = np.zeros((n,m,d))    
    
    u = U[:,:,0]
    v = U[:,:,1]
    for i in range(1,n-1):
        for j in range(1,m-1):
            LU[i,j,0] = laplace(u,i,j,dx,dy)
            LU[i,j,1] = laplace(v,i,j,dx,dy)
            
    return LU
            

def Df(U,dx,dy):
    n,m,d = np.shape(U)
    
    dUx = np.zeros((n,m))
    dUy = np.zeros((n,m))    
    dVx = np.zeros((n,m))
    dVy = np.zeros((n,m))    
    
    
    u = U[:,:,0]
    v = U[:,:,1]
    
    for i in range(1,n-1):
        for j in range(1,m-1):
            dUx[i,j] = (u[i+1,j]-u[i,j])/dx
            dUy[i,j] = (u[i,j+1]-u[i,j])/dy
            
            dVx[i,j] = (v[i+1,j]-v[i,j])/dx
            dVy[i,j] = (v[i,j+1]-v[i,j])/dy
            
    return dUx,dUy,dVx,dVy

def uGradU(U,dx,dy):
    dUx,dUy,dVx,dVy = Df(U,dx,dy)
    
    n,m,d = np.shape(U)
    G = np.zeros((n,m,d))
        
    for i in range(1,n-1):
        for j in range(1,m-1):
            
            G[i,j,0] = U[i,j,0]*dUx[i,j] + U[i,j,1]*dUy[i,j]
            G[i,j,1] = U[i,j,0]*dVx[i,j] + U[i,j,1]*dVy[i,j]
    
    return G
    


###############################################
a = 0.0
b = 5.0
N = 10

#viscosidade
nu = 0.5

#densidade
ro = 10.0

#parametros
dt = 0.5

h = (b-a)/(N)

dx = h
dy = h
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

###################################
#Prescrevendo a Fronteira

#Partes Superior e Inferior do Dominios 
#sera constante e igual a zero.

#Laterais esquerda e direita 
for i in range(1,numElemX-1):
    U[i,0] = [1.0,.0]
    U[i,numElemY-1] = [1.0,.0]

U1 = U
####################################
#matriz auxiliar
F =  np.zeros((numElemX,numElemY,2))

#matriz de pressao
P = np.zeros((numElemX,numElemY,2))
P[1:numElemX-1,1:numElemY-1,1] = np.ones((numElemX-2,numElemY-2)) 

####################################



nx = numElemX-1
ny = numElemY-1

for k in range(5):
    F = -uGradU(U,dx,dy) + nu*L(U,dx,dy) - (1.0/ro)*P
    U1[1:nx,1:ny,:] = U[1:nx,1:ny,:] + dt*F[1:nx,1:ny,:]
    print "*****", k , "****"
    print U1
    print "norma U nas bordas", np.linalg.norm(U1[1,0,:])
    print "\n"
    

    u = U[:,:,0]
    v = U[:,:,1]
    plt.figure()
    for i in range(numElemX):
        for j in range(numElemY):
            norma = np.linalg.norm(U1[i,j,:])
            plt.scatter(X[i,j],Y[i,j],c='r',s=70)
            #plt.text(X[i,j],Y[i,j],"%2.3f"%norma)
    Q = plt.quiver(X, Y, u, v)
    
    U = U1





plt.show()    