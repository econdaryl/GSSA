% This script solves Hansen's model with fixed labor
% It is intended as a example of how to implement the GSSA.m function

clear

% set Hansen model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m
global A gam bet del theta rho sig hbar
% options flags for GSSA
global numerSS
% parameters for GSSA
global nx ny nz NN SigZ


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
betaguess = [0; .95; .1; 0; 0; 0];
% betaguess = [0.318661593002810;    0.973912580282514;  0.367007250495616;...
%              0.000334776572529956; 0.0180101034589438; 0.202351767300829];
[beta,XYbar] = GSSA(XYbarguess,betaguess,'test');