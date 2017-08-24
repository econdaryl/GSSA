% This script solves Hansen's model with variable labor
% It is intended as a example of how to implement the GSSA.m function

clear

% set Hansen model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m
global A gam bet del theta rho sig mu B
% options flags for GSSA
global numerSS
% parameters for GSSA
global nx ny nz NN SigZ

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
nx = 1;          %number of endogenous state variables
ny = 1;          %number of jump variables
nz = 1;          %number of exogenous state variables
numerSS = 1;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.

%nz-by-nz matrix governing the law of motion for the exogenous state
%variables (called Z in GSSA function)
NN = rho;
%nz-by-nz matrix of variances/covariances for the exogenous state
%variables.
SigZ = sig;

% guess the steady state
XYbarguess = [10 .3];  

% run the GSSA proceedure
betaguess = [0.318661593002810    1;
             0.973912580282514    1; 
             0  0;
             0.367007250495616    1;
             0.000334776572529956  0;
             0  0;
             0.0180101034589438    0;
             0  0;
             0.202351767300829     0;
             0  0;];
[beta,XYbar] = GSSA(XYbarguess,betaguess,'test2')
