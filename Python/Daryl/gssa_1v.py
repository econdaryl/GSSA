# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:55:35 2017
@author: Daryl Larsen
"""
import numpy as np
import matplotlib.pyplot as plt
'''
Choose the simulation length
'''
T = 10000

'''
Model's parameters
'''
alpha = .35
beta = .99
gam = 2.5
delta = .08
chi = 10.
theta = 2.
tau = .05 # The first stochastic shock
rho = .9
sigma = .01
mparams = ([alpha, beta, gam, delta, chi, theta, tau, rho, sigma])
nx = 1
ny = 0
nz = 1
kbar = ((1-beta+beta*delta*(1-tau)) / (alpha*beta*(1-tau)))**(1/(alpha-1))

def Modeldefs(Xp, X, Z, params):
    kp = Xp
    k = X
    z = Z
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma] = params
    
    GDP = k**alpha*np.exp(z)
    r = alpha*GDP/k
    T = tau*(r - delta)*k
    c = (1-tau)*(r - delta)*k + k + T - kp
    i = GDP - c
    u = c**(1-gamma)/(1-gamma)
    return GDP, r, T, c, i, u

def Modeldyn(theta0, params):
    (Xpp, Xp, X, Zp, Z) = theta0
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma] = params
    
    GDP, r, T, c, i, u = Modeldefs(Xp, X, Z, params)
    GDPp, rp, Tp, cp, ip, up = Modeldefs(Xpp, Xp, Zp, params)
    
    #E1 = (c**(-gamma)*(1-tau)*w) / (chi) - 1
    E2 = (c**(-gamma)) / (beta*cp**(-gamma)*(1 + (1-tau)*(rp - delta))) - 1
    
    return np.array([E2])

def Modeldyn1(theta0, params):
    (Xpp, Xp, X, Zp, Z) = theta0
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma] = params
    
    GDP, w, r, T, c, i, u = Modeldefs(Xp, X, Z, params)
    GDPp, wp, rp, Tp, cp, ip, up = Modeldefs(Xpp, Xp, Zp, params)
    
    #E1 = (c**(-gamma)*(1-tau)*w) / (chi) - 1
    E2 = (beta*cp**(-gamma)*(1 + (1-tau)*(rp - delta))) / (c**(-gamma))
    
    return np.array([E2])

def poly1(Xin, XYparams):
    '''
    Includes polynomial terms up to order 'pord' for each element and quadratic 
    cross terms  One observation (row) at a time
    '''
    nX = nx + nz
    Xbasis = np.ones((1, 1))
    # generate polynomial terms for each element
    for i in range(1, pord):
        Xbasis = np.append(Xbasis, Xin**i)
    # generate cross terms
    for i in range (0, nX):
        for j in range(i+1, nX):
            temp = Xin[i]*Xin[j]
            Xbasis = np.append(Xbasis, temp)
    return Xbasis

def XYfunc(Xm, Zn, XYparams, coeffs):
    An = np.exp(Zn)
    XZin = np.append(Xm, An)
    XYbasis = np.append(1., XZin)
    for i in range(1, pord):
        XYbasis = poly1(XZin, XYparams)
    Xn = np.dot(XYbasis, coeffs)
    return Xn
    
def MVOLS(Y, X):
    '''
    OLS regression with observations in rows
    '''
    XX = np.dot(np.transpose(X), X)
    XY = np.dot(np.transpose(X), Y)
    coeffs = np.dot(np.linalg.inv(XX), XY)
    return coeffs
    
regtype = 'poly1' # functional form for X & Y functions 
fittype = 'MVOLS'   # regression fitting method
pord = 1  # order of polynomial for fitting function
ccrit = 1.0E-8  # convergence criteria for coeffs change
damp = 0.01  # damping paramter for fixed point algorithm
maxit = 500  # maximum number of iterations for fixed point algorithm
XYparams = (regtype, fittype, pord, nx, ny, nz, ccrit, damp)

# find model steady state
Zbar = np.zeros(nz)

# set up steady state input vector
theta0 = np.array([kbar, kbar, kbar, 0., 0.])

# check SS solution
check = Modeldyn(theta0, mparams)
print ('check SS: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')

#create history of Z's
Z = np.zeros([T,nz])
for t in range(1,T):
    Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma

''' First, with first-order polynomial (like in Matlab code) '''        
coeffs = np.array([0, 0.95, kbar*0.05]).reshape((3,1))
dist = 1.
count = 0
Xstart = np.ones(nx)*2.
Xold = np.ones((T, nx))

while dist > 1e-6 and count < maxit:
    count = count + 1
    X = np.zeros((T+1, nx))
    A = np.exp(Z)
    x = np.zeros((T,3))
    X[0] = XYfunc(Xstart, Z[0], XYparams, coeffs)
    for t in range(1,T+1):
        X[t] = XYfunc(X[t-1], Z[t-1], XYparams, coeffs)
        x[t-1] = np.concatenate((np.array([1.]), X[t-1], A[t-1]))
    X1 = X[0:T]
    # plot time series
    timeperiods = np.asarray(range(0,T))
    plt.plot(timeperiods, X1, label='X')
    plt.axhline(y=kbar, color='r')
    plt.title('time series')
    plt.xlabel('time')
    plt.legend(loc=9, ncol=(nx+ny))
    plt.show()    
    
    # Generate consumption and gamma time series
    c = (1-tau)*(alpha*X[0:T]**alpha*A-delta*X[0:T]) + X[0:T] + tau*(alpha*X[0:T]*A - delta*X[0:T]) - X[1:T+1]
    Gam = (beta*c[1:T]**(-gam)*(1-delta+alpha*X[1:T]**(alpha-1)*A[1:T]))/(c[0:T-1]**(-gam))
    
    # update values for X and Y
    Xnew = (Gam)*X[1:T]
    x = x[0:T-1,:]
    
    if fittype == 'MVOLS':
        coeffsnew = MVOLS(Xnew, x)
        
    # calculate distance between coeffs and coeffsnew
    diff = coeffs - coeffsnew
    print('coeffs', coeffs)
    print('coeffsnew', coeffsnew)
    print('X', X)
    
    #dist = np.max(np.abs(diff))  
    dist = np.mean(np.abs(1-X1/Xold))
    print('count ', count, 'distance', dist)
    
    # update coeffs
    Xold = X1
    coeffs = (1-damp)*coeffs + damp*coeffsnew
    
''' GSSA with higher-order polynomials '''
pord = 3
if regtype == 'poly1':
    cnumb = int(pord*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
    coeffs = np.ones((cnumb,(nx+ny)))*.1
    for i in range(0, nx+ny) :
        coeffs[:,i] = coeffs[:,i]*(i+1.)
dist = 1
count = 0
damp = 0.05

while dist > 1e-6 and count < maxit:
    count = count + 1
    X = np.zeros((T+1, nx))
    Xin = np.zeros((T, nx+nz))
    A = np.exp(Z)
    x = np.zeros((T,6))
    X[0] = XYfunc(Xstart, Z[0], XYparams, coeffs)
    for t in range(1,T+1):
        X[t] = XYfunc(X[t-1], Z[t-1], XYparams, coeffs)
        Xin[t-1,:] = np.concatenate((X[t-1], A[t-1]))
        x[t-1,:] = poly1(Xin[t-1,:], XYparams)
    X1 = X[0:T]
    # plot time series
    timeperiods = np.asarray(range(0,T))
    plt.plot(timeperiods, X1, label='X')
    plt.axhline(y=kbar, color='r')
    plt.title('time series')
    plt.xlabel('time')
    plt.legend(loc=9, ncol=(nx+ny))
    plt.show()    
    
    # Generate consumption and gamma series
    c = (1-tau)*(alpha*X[0:T]**alpha*A-delta*X[0:T]) + X[0:T] + tau*(alpha*X[0:T]*A - delta*X[0:T]) - X[1:T+1]
    Gam = (beta*c[1:T]**(-gam)*(1-delta+alpha*X[1:T]**(alpha-1)*A[1:T]))/(c[0:T-1]**(-gam))
    
    # update values for X and Y
    Xnew = (Gam)*X[1:T]
    x = x[0:T-1,:]
  
    if fittype == 'MVOLS':
        coeffsnew = MVOLS(Xnew, x)
        
    # calculate distance between coeffs and coeffsnew
    diff = coeffs - coeffsnew
    print('coeffs', coeffs)
    print('coeffsnew', coeffsnew)
    print('X', X)
    
    dist = np.mean(np.abs(1-X1/Xold))
    print('count ', count, 'distance', dist)
    
    # update coeffs
    Xold = X1
    coeffs = (1-damp)*coeffs + damp*coeffsnew