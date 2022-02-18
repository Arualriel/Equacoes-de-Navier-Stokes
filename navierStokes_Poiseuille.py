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
nu = 1.5 #agua

#densidade
ro = 1.0

#parametros
dt = 0.05

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
for i in range(1,numElemX-2):
    U[i,0] = [1.0,.0]
    U[i,numElemY-1] = [1.0,.0]

for j in range(0,numElemY-1):
	U[0,j] = [1.0,.0]
U1 = U
####################################
#matriz auxiliar
F =  np.zeros((numElemX,numElemY,2))

#matriz de pressao
dPx = -2.0##############################################derivada de p com relacao a x constante


P = np.zeros((numElemX,numElemY,2))

for i in range(numElemX):
    for j in range(numElemY):
        P[i,j,0] = dPx*x[i]  

####################################



nx = numElemX-1
ny = numElemY-1

for k in range(21):
    F= - (1.0/ro)*P
    #F = -uGradU(U,dx,dy) + nu*L(U,dx,dy) - (1.0/ro)*P #######################################
    U1[0:nx,0:ny,:] = U[0:nx,0:ny,:] + dt*F[0:nx,0:ny,:]
    print "*****", k , "****"
    #print U1
    #print "norma U nas bordas", np.linalg.norm(U1[1,0,:])
    #print "\n"
    
    u  = U[:,:,0]
    v  = U[:,:,1]

    u1 = U1[:,:,0]
    v1 = U1[:,:,1]


    if (k%5 == 0):


        plt.figure()
        for i in range(numElemX):
            for j in range(numElemY):
                norma = np.linalg.norm(U1[i,j,:])
                plt.scatter(X[i,j],Y[i,j],c='r',s=70)
                #plt.text(X[i,j],Y[i,j],"%2.3f"%norma)
        Q = plt.quiver(X, Y, u1, v1)
    
    normaU = np.linalg.norm(u-u1)
    normaV = np.linalg.norm(v-v1)
    
    print "Norma(u,u1) = %1.30f"%(normaU)     
    print "Norma(v,v1) = %1.30f"%(normaV)


    U[1:nx,1:ny,:] = U1[1:nx,1:ny,:]


#print "bordas laterais"
#print U[:,-1,:]
#print U[:,0,:]
#
#print "superior e inferior"
#print U[-1,:,:]
#print U[0,:,:]


plt.show()    
