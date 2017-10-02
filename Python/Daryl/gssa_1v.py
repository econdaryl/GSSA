# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:55:35 2017

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
    '''
    Given X and Z today generate X tomorrow
    '''
    # take natural logs of Xm 
    Xm = np.log(Xm)
    # concatenate Xm and Zn
    XZin = np.append(Xm, Zn)
    # choose from menu of functional forms
    if regtype == 'poly1':
        XYbasis = poly1(XZin, XYparams)
    XYout = np.dot(XYbasis, coeffs)
    Xn = XYout[0:nx]  
    # convert from logs to levels
    Xn = np.exp(Xn)
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
pord = 3  # order of polynomial for fitting function
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

# create history of Z's
Z = np.zeros([T,nz])
for t in range(1,T):
    Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sigma
# declare initial guess for coefficients
if regtype == 'poly1':
    cnumb = int(pord*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
    coeffs = np.ones((cnumb,(nx+ny)))*.1
    for i in range(0, nx+ny) :
        coeffs[:,i] = coeffs[:,i]*(i+1.)
        
#coeffs = [0, 0.95, kbar*0.05]
dist = 1.
count = 0
Xstart = np.ones(nx)*2

while dist > 1e-6:
    count = count + 1
    X = np.zeros((T, nx))
    
    X[0] = XYfunc(Xstart, Z[0], XYparams, coeffs)
    for t in range(1,T):
        X[t] = XYfunc(X[t-1], Z[t-1], XYparams, coeffs)
    # plot time series
    timeperiods = np.asarray(range(0,T))
    plt.plot(timeperiods, X, label='X')
    plt.title('time series')
    plt.xlabel('time')
    plt.legend(loc=9, ncol=(nx+ny))
    plt.show()    
    
    # initialize Gam and Lam series
    Gam = np.zeros((T,nx))
    Lam = np.zeros((T,ny))
    # generate discrete support for epsilon to be used in expectations
    # using rectangular arbitrage
    # Eps are the central values
    # Phi are the associated probabilities
    npts = 2
    Eps = np.zeros(npts);
    Cum = np.linspace(0.0, 1.0, num=npts+1)+.5/npts
    Cum = Cum[0:npts]
    Phi = np.ones(npts)/npts
    Eps = norm.ppf(Cum)
    
    # construct Gam and Lam series
    for i in range(0, npts):
        for j in range(0,npts):
            Xp = XYfunc(X[0], Z[0], XYparams, coeffs)
            Zp = Z[0] + np.array(Eps[i],Eps[j])
            theta01 = np.concatenate((Xp, X[0], Xstart, Zp, Z[0]),axis=0)
            tGam = Modeldyn(theta01, mparams) + 1
            Gam[0] = Gam[0] + tGam*Phi[i]*Phi[j]
    for t in range(1,T):
        for i in range(0,npts):
            for j in range(0,npts):
                Xp = XYfunc(X[t], Z[t], XYparams, coeffs)
                Zp = Z[t] + np.array(Eps[i],Eps[j])
                theta02 = np.concatenate((Xp, X[t], X[t-1], Zp, Z[t]),axis=0)
                tGam = Modeldyn(theta02, mparams) + 1
                Gam[t] = Gam[t] + tGam*Phi[i]*Phi[j]
    
    # update values for X and Y
    #Gam = np.abs(Gam)
    #Lam = np.abs(Lam)
    Xnew = (Gam)*X
    
    # run nonlinear regression to get new coefficients
    XZ = np.append(X, Z, axis = 1)
    XZ = XZ[0:T-1]
    Xnew = Xnew[1:T]
    if regtype == 'poly1':
        XZbasis = poly1(XZ[0], XYparams)
        XZbasis = np.atleast_2d(XZbasis)
        for t in range(1, T-1):
            temp = poly1(XZ[t], XYparams)
            temp = np.atleast_2d(temp)
            XZbasis = np.append(XZbasis, temp, axis=0)       
    if fittype == 'MVOLS':
        coeffsnew = MVOLS(Xnew, XZbasis)
        
    # calculate distance between coeffs and coeffsnew
    diff = coeffs - coeffsnew
    print('coeffs', coeffs)
    print('coeffsnew', coeffsnew)
    print('X', X)

    dist = np.max(np.abs(diff))  
    
    print('count ', count, 'distance', dist)
    
    # update coeffs
    coeffs = (1-damp)*coeffs + damp*coeffsnew
