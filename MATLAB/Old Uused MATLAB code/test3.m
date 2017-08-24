% This script solves a three-period OLG economy
% It is intended as a example of how to implement the GSSA.m function

clear

% set model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m
global A gam bet del alf rho sig
% options flags for GSSA
global numerSS
% parameters for GSSA
global nx ny nz NN SigZ

% set model parameters using same names as in test_dyn.m & test_def.m
A = 1;
alf = .36;
del = .025;
bet = .99;
gam = 1;
rho = .95;
sig = .0272;

% set GSSA parameters using same names as in GSSA
nx = 2;          %number of endogenous state variables
ny = 0;          %number of jump variables
nz = 1;          %number of exogenous state variables
numerSS = 1;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.

%nz-by-nz matrix governing the law of motion for the exogenous state
%variables (called Z in GSSA function)
NN = rho;
%nz-by-nz matrix of variances/covariances for the exogenous state
%variables.
SigZ = sig;

% guess the steady state
XYbarguess = [.1 .1];  

% run the GSSA proceedure
betaguess = [0.1  0.1;
             0.1  0.1; 
             0.9  0.1;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0];
[beta,XYbar] = GSSA(XYbarguess,betaguess,'test3')