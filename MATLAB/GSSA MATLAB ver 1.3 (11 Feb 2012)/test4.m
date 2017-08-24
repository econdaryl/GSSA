% This script solves a unstable fiscal policy model
% It is intended as a example of how to implement the GSSA.m function

clear

% set model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m
global gamma bet delta alpha rhoz sigz lbar taubar d 
global Bbase Bmax Bupp Blow Bmin 
% options flags for GSSA
global numerSS
% parameters for GSSA
global nx ny nz NN SigZ

% set model parameters using same names as in test_dyn.m & test_def.m
alpha = .3;   %capital share in GDP .3
bet = .995;  %discount factor .995
delta = .025; %depreciation rate 100% for exact solution to hold
lbar = 1;     %labor per worker  1
gamma = 2.0;  %utility curvature
taubar = .15; %tax rate
d = .25;        %transfer
sigz = .02;   %standard deviation for shocks .02
rhoz = .99;   %autocorrelation of z series .9
Bbase = 20;
Bmax = 2*Bbase;
Bupp = Bbase;
Blow = -Bbase;
Bmin = -2*Bbase;
rbar1 = 1/(bet*(1-taubar))-1+delta

% set GSSA parameters using same names as in GSSA
nx = 2;          %number of endogenous state variables
ny = 0;          %number of jump variables
nz = 1;          %number of exogenous state variables
numerSS = 1;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.

%nz-by-nz matrix governing the law of motion for the exogenous state
%variables (called Z in GSSA function)
NN = rhoz;
%nz-by-nz matrix of variances/covariances for the exogenous state
%variables.
SigZ = sigz;

% guess the steady state
XYbarguess = [.1 .1];  

% run the GSSA proceedure
betaguess = [26.7949   33.0780;
             -0.0  0.0;
             0.0  -0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0;
             0.0  0.0];
[beta,XYbar] = GSSA(XYbarguess,betaguess,'test4')