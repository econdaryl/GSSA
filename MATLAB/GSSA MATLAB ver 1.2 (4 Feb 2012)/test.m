% This script solves Hansen's model with fixed labor
% It is intended as a example of how to implement the GSSA.m function

clear

% set Hansen model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m
global A gam bet del theta rho sig hbar
% options flags for GSSA
global numerSS fittype
% parameters for GSSA
global nx ny nz npar NN SigZ


% set model parameters using same names as in test_dyn.m & test_def.m
A = 1;
theta = .33; %(.33)
del = .025;  %(.025)
bet = .995;  %(.995)
hbar = .3;   %(.3)
gam = 1;     %(2)
rho = .9;    %(.9)
sig = .02;   %(.02)

% set GSSA parameters using same names as in GSSA
nx = 1;          %number of endogenous state variables
ny = 0;          %number of jump variables
nz = 1;          %number of exogenous state variables
numerSS = 0;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.
fittype = 3;     %functional form for approximations
                 %0 is linear (AVOID)
                 %1 is log-linear
                 %2 is quadratic (AVOID)
                 %3 is log-quadratic

%nz-by-nz matrix governing the law of motion for the exogenous state
%variables (called Z in GSSA function)
NN = rho;
%nz-by-nz matrix of variances/covariances for the exogenous state
%variables.
SigZ = sig;

% find the steady state algebraically
zbar = 0;
kbar = hbar*((theta*A)/(1/bet+del-1))^(1/(1-theta));
[ybar, ibar, cbar, rbar, wbar] = test_def(kbar,zbar,kbar);
bar = [kbar; ybar; ibar; cbar; rbar; wbar];
disp(bar')

%set up a row vector guess at the steady state values of the X & Y
%variables.  We will use the algebraic solution.
XYbarguess = kbar;  

% run the GSSA proceedure
AA = [];
BB = [.966; .052];
CC = [];
[XYpars,XYbar,errors] = GSSA(XYbarguess,AA,BB,CC,'test');

% reshape the output into appropriate matrices
beta = reshape(XYpars,npar,nx+ny);
[AA,BB,CC] = GSSA_fittype(beta)

errors