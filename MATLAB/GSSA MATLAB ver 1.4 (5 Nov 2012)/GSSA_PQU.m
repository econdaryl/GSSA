% GSSA package
% version 1.4 written by Kerk L. Phillips  11/5/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables. 

function [PP,QQ,UU,RR,SS,VV] = GSSA_PQU(theta0, Zbar, GSSAparams,...
    modelparam)
global dyneqns
% This function solves for the linear coefficient matrices of poolicy and 
% jump variable functions
%
% This function takes 4 inputs:
% 1) theta0, the vector [X(t+2),X(t+1),X(t),Y(t+1),Y(t),Z(t+1),Z(t)]
% 2) Zbar, steady state value for Z
% 3) GSSAparams, the vector of paramter values from the GSSA function
% 4) modelparams, a vector of model specific parameter values passed to the
%    model dynamic function named dyneqns
%
% The output is the coeffient matrices for the following approximate policy
% and jump variable functions.
%  X(t+1) = UU + X(t)*PP + Z(t)*QQ
%  X(t+1) = VV + X(t)*RR + Z(t)*SS
%
% It requires the following subroutines written by by Kerk L. Phillips
% incorporating code written by Harald Uhlig.
% 1) LinApp_Deriv.m - takes numerical derivatives
% 2) LinApp_Solve.m - solves for the linear coefficients

% read in GSSA parameters
nx           = GSSAparams(1);
ny           = GSSAparams(2);
nz           = GSSAparams(3);
fittype      = GSSAparams(10);
NN           = GSSAparams(17);
if fittype == 2
    logX = 1;
else
    logX = 0;
end


modelparam
theta0
[nx,ny,nz]
logX
% rename the state about which the approximation will be made
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = ...
    LinApp_Deriv(dyneqns,modelparam,theta0,nx,ny,nz,logX);
Z0 = theta0(3*nx+2*ny+1,:)-Zbar;

% find coefficients
[PP, QQ, UU, RR, SS, VV] = ...
  LinApp_Solve(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT, NN, Z0);