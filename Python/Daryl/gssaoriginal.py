# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:59:06 2017

@author: Daryl Larsen
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 08 12:53:03 2017
@author: Daryl Larsen
"""

'''
This program implements the Generalized Stochastic Simulation Algorthim from 
Judd, Maliar and Mailar (2011) "Numerically stable and accurate stochastic 
simulation approaches for solving dynamic economic models", Quantitative
Economics vol. 2, pp. 173-210.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as opt

from LinApp_FindSS import LinApp_FindSS

'''
We test the algorithm with a simple DSGE model with endogenous labor.
'''

def modeldefs(Xn, Xm, Yn, Zn, mparams):
    '''
    Definitions functions for the model
    '''
    K = np.sum(Xm)
    L = Yn*np.exp(Zn)
    Y = K**alpha*L**(1-alpha)
    r = alpha*Y/K
    w = (Y - r*K)/L
    T = tau*(w*L + (r-delta)*K)
    C =(1-tau)*(w*L +(r-delta)*Xm) + Xm + T - Xn
    u = 1/(1-sig)*(C**(1-sig)-1)-chi*(1/(1+theta))*L**(1+theta)
    return C, w, r, Y, K, L, u
    
    
def modeldyn(Xp, Xn, Xm, Yp, Yn, Zp, Zn, mparams):
    '''
    Dynamic Euler equations for the model
    X and Y are input as natural logs
    '''
    Cn, wn, rn, Yn, Km, Ln, un = modeldefs(Xn, Xm, Yn, Zn, mparams)
    Cp, wp, rp, Yp, Kn, L2, up = modeldefs(Xp, Xn, Yp, Zp, mparams)
    Gamma =  (beta*Cp**(-gamma)*(1+rp-delta))/(Cn**(-gamma))
    Lambda = -(Cn**(-gamma)*wn)/(chi*Ln**theta)
    # print Gamma, Lambda
    return Gamma, Lambda
    
    
def modelss(XYbar, Zbar, mparams):
    # unpack bar
    Xbar = XYbar[0:nx]
    Ybar = XYbar[nx:nx+ny]
    Gambar, Lambar = modeldyn(Xbar, Xbar, Xbar, Ybar, Ybar, Zbar, Zbar, \
        mparams)
    return np.append(Gambar - 1.,Lambar - 1.)
    

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
            temp =Xin[i]*Xin[j]
            Xbasis = np.append(Xbasis, temp)
    return Xbasis
    
    
def XYfunc(Xm, Zn, XYparams, coeffs):
    '''
    Given X and Z today generate X and Y tomorrow
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
    Yn = XYout[nx:nx+ny]  
    # convert from logs to levels
    Xn = np.exp(Xn)  
    Yn = np.exp(Yn)    
    return Xn, Yn
    
    
def MVOLS(Y, X):
    '''
    OLS regression with observations in rows
    '''
    XX = np.dot(np.transpose(X), X)
    XY = np.dot(np.transpose(X), Y)
    coeffs = np.dot(np.linalg.inv(XX), XY)
    return coeffs
    

# main program

# declare model parameters
alpha = .33  # capital share
beta = .95   # discount factor
gamma = 2.5  # CES value
delta = .10  # depreciation rate
theta = 2.0  # constant Frisch elasticity
tau = 0.05   # tax rate
chi = 5.     # labor disutility weight
sig = .02    # square-rrot of vcv matrix for Z VAR
rho = .9     # VAR matrix for Z
nx = 2
ny = 2
nz = 1

NN = rho  # VAR matrix for stochastic shocks
Sig = sig # VCV matrix
Sigsr = np.sqrt(Sig) # square root of VCV matrix
mparams = (alpha, beta, gamma, delta, theta, chi, tau, NN)

# declare other parameters
T = 20   # number of observations to simulate
regtype = 'poly1' # functional form for X & Y functions 
fittype = 'MVOLS'   # regression fitting method
pord = 3  # order of polynomial for fitting function
ccrit = 1.0E-8  # convergence criteria for coeffs change
damp = 1.   # damping paramter for fixed point algorithm
maxit = 500  # maximum number of iterations for fixed point algorithm
XYparams = (regtype, fittype, pord, nx, ny, nz, ccrit, damp)

#############################################################
# find model steady state
# set up lambda funtion for fsolve
Zbar = np.zeros(nz)
guessXY = np.ones(nx+ny)*.01
f = lambda XYbar: modelss(XYbar, Zbar, mparams)
XYbar = opt.fsolve(f, x0=guessXY)
checkbar = modelss(XYbar, Zbar, mparams) # Find steady state code from Anthony and LinApp_FindSS
Xbar = XYbar[0:nx]
Ybar = XYbar[nx:nx+ny]
print('Xbar: ', Xbar)
print('Ybar: ', Ybar)
print('checkbar: ', checkbar)

# create history of Z's
'''
Z = np.zeros([T,nz])
for t in range(1, T):
    Z[t,:] =  np.dot(np.random.randn(1,nz), Sigsr) + np.dot(Z[t-1,:], NN)
'''
Z = np.zeros([T,nz])
for t in range(1,T):
    Z[t,:] = rho*Z[t-1] + np.random.randn(1)*sig
# declare initial guess for coefficients
if regtype == 'poly1':
    cnumb = int(pord*(nx+nz) + .5*(nx+nz-1)*(nx+nz-2))
    coeffs = np.ones((cnumb,(nx+ny)))*.01
    for i in range(0, nx+ny) :
        coeffs[:,i] = coeffs[:,i]*(i+1.)
    
# declare starting values
Xstart = np.ones((1,nx))*2

# begin fixed point iteration
dist = 1.
count = 0
while dist > ccrit:
    # check for maximum number of iterations
    count = count + 1
    if count > maxit:
        break
    # initialize X and Y series
    X = np.zeros((T,nx))
    Y = np.zeros((T,ny))
    
    # constuct X and Y series
    X[0,:], Y[0,:] = XYfunc(Xstart, Z[0,:], XYparams, coeffs)
    for t in range(1,T):
        X[t,:], Y[t,:] = XYfunc(X[t-1,:], Z[t-1,:], XYparams, coeffs)
    
    # plot time series
    timeperiods = np.asarray(range(0,T))
    plt.plot(timeperiods, X, label='X')
    plt.plot(timeperiods, Y, label='Y')
    plt.title('time series')
    plt.xlabel('time')
    plt.legend(loc=9, ncol=(nx+ny))
    plt.show()    
    
    # initialize Gam and Lam series
    Gam = np.zeros((T,nx))
    Lam = np.zeros((T,ny))
    
    # generate discret support for epsilon to be used in expectations
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
        for j in range(0, npts):
            Zp = Z[0,:] + np.array(Eps[i],Eps[j])
            Xp, Yp = XYfunc(X[0,:], Zp, XYparams, coeffs)
            tGam, tLam = modeldyn(Xp, X[0,:], Xstart, Yp, Y[0,:], Zp, Z[0], \
                mparams)
            Gam[0] = Gam[0] + tGam*Phi[i]*Phi[j]
            Lam[0] = Lam[0] + tLam*Phi[i]*Phi[j]
    for t in range(1,T):
        for i in range(0, npts):
            for j in range(0, npts):
                Zp = Z[t,:] + np.array(Eps[i],Eps[j])
                Xp, Yp = XYfunc(X[t,:], Zp, XYparams, coeffs)
                tGam, tLam = modeldyn(Xp, X[t,:], X[t-1,:], Yp, Y[t,:], Zp, Z[t], \
                    mparams)
                Gam[t] = Gam[t] + tGam*Phi[i]*Phi[j]
                Lam[t] = Lam[t] + tLam*Phi[i]*Phi[j]
    
    # update values for X and Y
    Xnew = Gam*X
    Ynew = Lam*Y
    
    #print 'Xnew: ', Xnew
    #print 'Ynew: ', Ynew
    
    # run nonlinear regression to get new coefficients
    XZ = np.append(X, Z, axis = 1)
    XY = np.append(X, Y, axis = 1)
    XZ = XZ[0:T-1,:]
    XY = XY[1:T,:]
    if regtype == 'poly1':
        XZbasis = poly1(XZ[0], XYparams)
        XZbasis = np.atleast_2d(XZbasis)
        for t in range(1, T-1):
            temp = poly1(XZ[t], XYparams)
            temp = np.atleast_2d(temp)
            XZbasis = np.append(XZbasis, temp, axis=0)       
    if fittype == 'MVOLS':
        coeffsnew = MVOLS(XY, XZbasis)
        
    # calculate distance between coeffs and coeffsnew
    diff = coeffs - coeffsnew
    print('coeffs', coeffs)
    print('coeffsnew', coeffsnew)
    print('X', X)
    print('Y', Y)
    if np.isnan(coeffsnew).any():
        print('wtf')
    dist = np.max(np.abs(diff))  
    
    print('count ', count, 'distance', dist)
    
    # update coeffs
    coeffs = (1-damp)*coeffs + damp*coeffsnew