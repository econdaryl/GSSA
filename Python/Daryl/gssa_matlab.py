# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:05:13 2017

@author: Daryl Larsen
"""

import numpy as np
#from LinApp_FindSS import LinApp_FindSS
import matplotlib.pyplot as plt
from scipy.stats import norm
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
kbar = ((1-beta+beta*delta) / (alpha*beta))**(1/(alpha-1))

def Modeldefs(Xp, X, Z, params):
    kp = Xp
    k = X
    z = Z
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma] = params
    
    Y = k**alpha*np.exp(z)
    w = (1-alpha)*Y
    r = alpha*Y/k
    T = tau*(w + (r - delta)*k)
    c = (1-tau)*(w + (r - delta)*k) + k + T - kp
    i = Y - c
    u = c**(1-gamma)/(1-gamma)
    return Y, w, r, T, c, i, u

def Modeldyn(theta0, params):
    (Xpp, Xp, X, Zp, Z) = theta0
    
    [alpha, beta, gamma, delta, chi, theta, tau, rho, sigma] = params
    
    Y, w, r, T, c, i, u = Modeldefs(Xp, X, Z, params)
    Yp, wp, rp, Tp, cp, ip, up = Modeldefs(Xpp, Xp, Zp, params)
    
    #E1 = (c**(-gamma)*(1-tau)*w) / (chi) - 1
    E2 = (c**(-gamma)) / (beta*cp**(-gamma)*(1 + (1-tau)*(rp - delta))) - 1
    
    return np.array([E2])

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
pord = 3  # order of polynomial for fitting function
ccrit = 1.0E-8  # convergence criteria for coeffs change
damp = 0.05  # damping paramter for fixed point algorithm
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

# create history of Z's
Z = np.zeros([T,nz])
for t in range(1,T):
    Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
# declare initial guess for coefficients
#if regtype == 'poly1':
#    cnumb = int(pord*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
#    coeffs = np.ones((cnumb,(nx+ny)))*.1
#    for i in range(0, nx+ny) :
#        coeffs[:,i] = coeffs[:,i]*(i+1.)
        
coeffs = np.array([0., 0.95, 0.05*kbar]).reshape(3,1)

dist = 1.
maxit = 500
count = 0

Xstart = np.ones(nx)*kbar
k = np.ones((T+1,1))*kbar
k_old = k
X = np.zeros((T,3))
Y = np.zeros((T,1))
w = np.zeros((T,1))
r = np.zeros((T,1))
Tax = np.zeros((T,1))
c = np.zeros((T,1))
i = np.zeros((T,1))
u = np.zeros((T,1))

while dist > 1e-6 and count < maxit:
    count = count+1
    for t in range(T):
        X[t, :] = [1, k[t], np.exp(Z[t])]
        k[t+1] = np.dot(X[t], coeffs)
        Y[t],w[t],r[t],Tax[t],c[t],i[t],u[t] = Modeldefs(k[t], k[t+1], Z[t], mparams)
    y = np.linalg.matrix_power(beta*c[2:T,1],(-gam)) / np.linalg.matrix_power(c[1:T-1,1],(-gam))*(1-delta+alpha*np.linalg.matrix_power(k[2:T,1], (alpha-1))*np.exp(Z[2:T,1]))*k[2:T,1]
    k1 = k[0:T]
    timeperiods = np.asarray(range(0,T))
    plt.plot(timeperiods, k1, label='X')
    plt.title('time series')
    plt.xlabel('time')
    plt.legend(loc=9, ncol=(nx+ny))
    plt.show() 
    coeffsnew = MVOLS(Y, X)
    #dist = np.max(np.abs(1-k./k_old))
    diff = coeffs - coeffsnew
    print('coeffs', coeffs)
    print('coeffsnew', coeffsnew)
    print('X', X)

    dist = np.max(np.abs(diff))  
    
    print('count ', count, 'distance', dist)
    
    coeffs = (1-damp)*coeffs + damp*coeffsnew
    