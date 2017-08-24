% This script solves Hansen's model with fixed labor
% It is intended as a example of how to implement the GSSA.m function

clear

% set Hansen model parameters and set neccessary global variables.
% Not used by GSSA, but used by test_dyn.m & test_def.m

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
nx = 1;          % number of endogenous state variables
ny = 0;          % number of jump variables
nz = 1;          % number of exogenous state variables
numerSS = 0;     % set to 1 if XYbarguess is a guess, 0 if it is exact SS.
T = 1000;        % number of observations in the simulation sample
kdamp = 0.05;    % Damping parameter for (fixed-point) iteration on 
                 % the coefficients of the capital policy functions
maxwhile = 1000; % maximimum iterations for approximations
usePQ = 0;       % 1 to use a linear approximation to get initial guess 
                 % for beta
dotrunc = 0;     % 1 to truncate data outside constraints, 0 otherwise
fittype = 1;     % functional form for polynomial approximations
                 % 1) linear
                 % 2) log-linear
RM = 5;          % regression (approximation) method, RM=1,...,8:  
                 % 1=OLS,          2=LS-SVD,    3=LAD-PP,    4=LAD-DP, 
                 % 5=RLS-Tikhonov, 6=RLS-TSVD,  7=RLAD-PP,   8=RLAD-DP;
penalty = 7;     % a parameter determining the value of the regularization
                 % parameter for a regularization methods, RM=5,6,7,8;
D = 5;           % order of polynomical approximation (1,2,3,4 or 5)
PF = 2;          % polynomial family
                 % 1) ordinary
                 % 2) Hermite
quadtype = 4;    % type of quadrature used
                 % 1) "Monomials_1.m"      
                 % constructs integration nodes and weights for an N-
                 % dimensional monomial (non-product) integration rule 
                 % with 2N nodes 
                 % 2) "Monomials_2.m"      
                 % constructs integration nodes and weights for an N-
                 % dimensional monomial (non-product) integration rule 
                 % with 2N^2+1 nodes
                 % 3)"GH_Quadrature.m"    
                 % constructs integration nodes and weights for the
                 % Gauss-Hermite rules with the number of nodes in 
                 % each dimension ranging from one to ten  
                 % 4)"GSSA_nodes.m"
                 % does rectangular arbitrage for a large number of nodes
                 % used only for univariate shock case.
Qn = 10;         % the number of nodes in each dimension; 1<=Qn<=10
                 % for "GH_Quadrature.m" only
NN = rho;        % nz-by-nz matrix governing the law of motion for the 
                 % exogenous state variables (called Z in GSSA function)
SigZ = sig;      % nz-by-nz matrix of variances/covariances for the
                 % exogenous state variables.
X1 = -999;       % nx-by-1 vector of starting values
                 % setting this to -999 uses the steady state as starting
                 % values
modelparams = [A, theta, del, bet, hbar, gam, rho, sig];
GSSAparams =  [nx, ny, nz, numerSS, T, kdamp, maxwhile, usePQ, dotrunc,...
               fittype, RM, penalty, D, PF, quadtype, Qn, NN, SigZ, X1];

% find the steady state algebraically
zbar = 0;
kbar = hbar*((theta*A)/(1/bet+del-1))^(1/(1-theta));
[ybar, ibar, cbar, rbar, wbar] = test_def(kbar,zbar,kbar,modelparams);
bar = [kbar; ybar; ibar; cbar; rbar; wbar];
disp(bar')

%set up a row vector guess at the steady state values of the X & Y
%variables.  We will use the algebraic solution.
XYbarguess = kbar;  

% run the GSSA proceedure
betaguess = [0; .95; .1; 0; 0; 0];
% betaguess = [0.318661593002810;    0.973912580282514;  0.367007250495616;...
%              0.000334776572529956; 0.0180101034589438; 0.202351767300829];
[beta,XYbar,eulerr] = GSSA(XYbarguess, betaguess, 'test', GSSAparams,...
    modelparams)