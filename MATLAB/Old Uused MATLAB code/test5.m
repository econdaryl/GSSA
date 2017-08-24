% This script solves a model with discrete marginal tax rates
% It is intended as a example of how to implement the GSSA.m function

clear

% set Hansen model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m
global alpha beta delta gamma lbar sigz rhoz tau1 tau2 taxthreshold
% options flags for GSSA
global numerSS
% parameters for GSSA
global nx ny nz NN SigZ

% set model parameters
alpha = .3;   %capital share in GDP .3
beta = .995;  %discount factor .995
delta = .025; %depreciation rate 100% for exact solution to hold
lbar = 1;     %labor per worker  1
gamma = 1.0;  %utility curvature
tau1 = .0;     %lower tax rate
tau2 = tau1*1;   %higher tax rate
taxthreshold = 1.96;
sigz = .02;   %standard deviation for shocks .02
rhoz = .9;    %autocorrelation of z series .9
rbar1 = 1/(beta*(1-tau1))-1+delta

% set GSSA parameters using same names as in GSSA
nx = 1;          %number of endogenous state variables
ny = 0;          %number of jump variables
nz = 1;          %number of exogenous state variables
numerSS = 1;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.

%nz-by-nz matrix governing the law of motion for the exogenous state
%variables (called Z in GSSA function)
NN = rhoz;
%nz-by-nz matrix of variances/covariances for the exogenous state
%variables.
SigZ = sigz;

%set up a row vector guess at the steady state values of the X & Y
%variables.  We will use the algebraic solution.
XYbarguess = 24;  

% run the GSSA proceedure
betaguess = [0; 0.95; 0.1; 0; 0; 0];
[beta,XYbar] = GSSA(XYbarguess,betaguess,'test5');