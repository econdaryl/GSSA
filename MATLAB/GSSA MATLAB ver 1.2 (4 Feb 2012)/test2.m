% This script solves Hansen's model with variable labor
% It is intended as a example of how to implement the GSSA.m function

clear

% set Hansen model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m
global A gam bet del theta rho sig mu B
% options flags for GSSA
global numerSS fittype
% parameters for GSSA
global nx ny nz betar NN SigZ

% set model parameters using same names as in test_dyn.m & test_def.m
A = 1;
theta = .36;
del = .025;
bet = .99;
gam = 1;
mu = 1;
B = 1.72;
rho = .95;
sig = .0272;

% set GSSA parameters using same names as in GSSA
nx = 2;          %number of endogenous state variables
ny = 0;          %number of jump variables
nz = 1;          %number of exogenous state variables
numerSS = 1;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.
fittype = 1;     %functional form for approximations
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

% guess the steady state
XYbarguess = [10 .3];  

% run the GSSA proceedure
AA = [];
BB = [.966 .01; .01 .75; .052 .01];
CC = [];
[XYpars,XYbar,errors] = GSSA(XYbarguess,AA,BB,CC,'test2');

% reshape the output into appropriate matrices
beta = reshape(XYpars,betar,nx+ny);
[AA,BB,CC] = GSSA_fittype(beta)

errors