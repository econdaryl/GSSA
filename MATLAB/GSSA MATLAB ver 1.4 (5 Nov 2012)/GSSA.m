% GSSA package
% version 1.4 written by Kerk L. Phillips  11/5/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables.         

function [beta,XYbar,eulerr] = GSSA(XYbarguess, beta, modelname,...
    GSSAparams, modelparams)
global nx ny nz dyneqns trunceqns
% This function takes three inputs:
% 1) XYbarguess
%    a guess for the steady state value of the endogneous state variables &
%    jump variables, XYbarguess  It outputs the parameters for an
%    approximation of the state transition function, X(t+1) = f(X(t),Z(t)) 
%    and the jump variable function Y(t) = g(X(t),Z(t)).
%    where X is a vector of nx endogenous state variables
%    Y is a vector of ny endogenous state variables
%    and Z is a vector of nz exogenous state variables
% 2) beta
%    an initial guess for the parameter matrix
% 3) modelname
%    a string specifying the model's name
% 4) GSSAparams
%    a vector of GSSA parameters
% 5) modelparams
%    a vectpr of model specific parameters that is passed to the
%    appropriate "modelname"_dyn and "modelname"_trunc files.
%
% The output is:
% 1) beta, a vector of approximation coeffcients.
% 2) XYbar, a vector of steady state values for X & Y.
% 3) eulerr
%
% The GSSA parameters to be set are:
% nx       number of endogenous state variables
% ny       number of jump variables
% nz       number of exogenous state variables
% numerSS  set to 1 if XYbarguess is a guess, 0 if it is exact SS.
% T        number of observations in the simulation sample
% kdamp    Damping parameter for (fixed-point) iteration onthe coefficients
%          of the capital policy functions
% maxwhile maximimum iterations for approximations
% usePQ    1 to use a linear approximation to get initial guess for beta
% dotrunc  1 to truncate data outside constraints, 0 otherwise
% fittype  functional form for polynomial approximations
%           1) linear
%           2) log-linear
% RM       regression (approximation) method, RM=1,...,8:  
%           1=OLS,          2=LS-SVD,    3=LAD-PP,    4=LAD-DP, 
%           5=RLS-Tikhonov, 6=RLS-TSVD,  7=RLAD-PP,   8=RLAD-DP;
% penalty  a parameter determining the value of the regularization
%          parameter for a regularization methods, RM=5,6,7,8;
% D        order of polynomical approximation (1,2,3,4 or 5)
% PF       polynomial family
%           1) ordinary
%           2) Hermite
% quadtype type of quadrature used
%           1) "Monomials_1.m" - constructs integration nodes and weights 
%              for an N-dimensional monomial (non-product) integration rule 
%              with 2N nodes 
%           2) "Monomials_2.m" - constructs integration nodes and weights 
%              for an N-dimensional monomial (non-product) integration rule 
%              with 2N^2+1 nodes
%           3) "GH_Quadrature.m" - constructs integration nodes and weights 
%              for the Gauss-Hermite rules with the number of nodes in each
%              dimension ranging from one to ten  
%           4) "GSSA_nodes.m" - does rectangular arbitrage for a large 
%              number of nodesused only for univariate shock case.
% Qn       the number of nodes in each dimension; 1<=Qn<=10 for 
%          "GH_Quadrature.m" only
% SigZ     z-by-nz matrix of variances/covariances for the
%          exogenous state variables.
% X1       nx-by-1 vector of starting values
%          setting this to -999 uses the steady state as starting values
%
% It requires the following subroutines written by by Kenneth L. Judd, 
% Lilia Maliar and Serguei Maliar which are available as a zip file from
% Lilia & Setguei Mailar's webapage at:
% http://www.stanford.edu/~maliars/Files/Codes.html
% 1. "Num_Stab_Approx.m"  implements the numerically stable LS and LAD 
%                         approximation methods
% 2. "Ord_Polynomial_N.m" constructs the sets of basis functions for ordinary 
%                         polynomials of the degrees from one to five, for
%                         the N-country model
% 3. "Monomials_1.m"      constructs integration nodes and weights for an N-
%                         dimensional monomial (non-product) integration rule 
%                         with 2N nodes 
% 4. "Monomials_2.m"      constructs integration nodes and weights for an N-
%                         dimensional monomial (non-product) integration rule 
%                         with 2N^2+1 nodes
% 5. "GH_Quadrature.m"    constructs integration nodes and weights for the 
%                         Gauss-Hermite rules with the number of nodes in 
%                         each dimension ranging from one to ten      
%
% It also requires one model-specific external function:
%  It should be named "modelname"_dyn.m and should take as inputs:
%    X(t+2), X(t+1), X(t), Y(t+1), Y(t), Z(t+1) & Z(t)
%  It should output a with nx+ny elements column vector with the values of 
%  the model's behavioral equations (many of which will be Euler equations)
%  written in such a way that the equation evaluates to zero.  This 
%  function is used to find the steady state and in the GSSA algorithm 
%  itself.
% An optional file named "modelname"_trunc.m takes the XY vector in a given
%  period and truncates the variables to any upper and lower bounds set in
%  that function.

% read in GSSA parameters
nx       = GSSAparams(1);
ny       = GSSAparams(2);
nz       = GSSAparams(3);
numerSS  = GSSAparams(4);
T        = GSSAparams(5);
kdamp    = GSSAparams(6);
maxwhile = GSSAparams(7);
usePQ    = GSSAparams(8);
dotrunc  = GSSAparams(9);
fittype  = GSSAparams(10);
RM       = GSSAparams(11);
penalty  = GSSAparams(12);
D        = GSSAparams(13);
PF       = GSSAparams(14);
quadtype = GSSAparams(15);
Qn       = GSSAparams(16);
NN       = GSSAparams(17);
SigZ     = GSSAparams(18);
X1       = GSSAparams(19);

% create the name of the dynamic behavioral equations function
%  fundamental dynamic equations that define the model
dyneqns = str2func([modelname '_dyn']);
%  restrictions on values of variables during simulation
trunceqns = str2func([modelname '_trunc']);
          
% calculate nodes and weights for quadrature
% Inputs:  "nz" is the number of random variables; N>=1;
%          "SigZ" is the variance-covariance matrix; N-by-N
% Outputs: "n_nodes" is the total number of integration nodes; 2*N;
%          "epsi_nodes" are the integration nodes; n_nodes-by-N;
%          "weight_nodes" are the integration weights; n_nodes-by-1
if quadtype == 1
    [J,epsi_nodes,weight_nodes] = Monomials_1(nz,SigZ);
elseif quadtype == 2
    [J,epsi_nodes,weight_nodes] = Monomials_2(nz,SigZ);
elseif quadtype == 3
    % Inputs:  "Qn" is the number of nodes in each dimension; 1<=Qn<=10;
    [J,epsi_nodes,weight_nodes] = GH_Quadrature(Qn,nz,SigZ);
elseif quadtype == 4
    J = 20;
    [epsi_nodes, weight_nodes] = GSSA_nodes(J,nz,SigZ);
end

% find steady state numerically; skip if you already have exact values
if numerSS == 1
    options = optimset('Display','iter');
    XYbar = fsolve(@GSSA_ss,XYbarguess,options);
else
    XYbar = XYbarguess;
end
Xbar = XYbar(1:nx);
Ybar = XYbar(nx+1:nx+ny);
Zbar = zeros(nz,1);

% generate a random history of Z's to be used throughout
Z = zeros(T,nz);
eps = randn(T,nz)*SigZ;
for t=2:T
    Z(t,:) = Z(t-1,:)*NN + eps(t,:);
end

if usePQ == 1
    % get an initial estimate of beta by simulating about the steady state
    % using Uhlig's solution method.
    in = [Xbar; Xbar; Xbar; Ybar; Ybar; Zbar; Zbar];
    [PP,QQ,UU,RR,SS,VV] = GSSA_PQU(in,Zbar,GSSAparams,modelparams);

    % generate X & Y data given Z's above
    Xtilde = zeros(T,nx);
    X = ones(T,nx);
    if ny> 0
        Ytilde = zeros(T,ny);
        Y = ones(T,ny);
    end
    X(1,:) = Xbar;
    if ny > 0
       Y(1,:) = Ybar;
    end
    for t=2:T
        Xtilde(t,:) = UU + Xtilde(t-1,:)*PP' + Z(t-1,:)*QQ';
        X(t,:) = Xbar.*exp(Xtilde(t,:));
        if ny > 0
            Ytilde(t,:) = VV + Xtilde(t-1,:)*RR' + Z(t-1,:)*SS';
            Y(t,:) = Ybar.*exp(Ytilde(t,:));
        end
    end
    Xtildefinal = UU + Xtilde(T,:)*PP' + Z(T,:)*QQ';
    Xfinal = Xbar.*exp(Xtildefinal);
    if ny > 0
        Ytildefinal = VV + Ytilde(T,:)*RR' + Z(T,:)*SS';
        Yfinal = Ybar.*exp(Ytildefinal);
    end
    
    % estimate beta using this data
    if ny > 0
        Xrhs = [X Y];
        XYlhs = [X(2:T,:) Y(2:T,:); Xfinal Yfinal];
    else
        Xrhs = X;
        XYlhs = [X(2:T,:); Xfinal];
    end
   
    if fittype == 2
        Xrhs = log(Xrhs);
        XYlhs = log(XYlhs);
    end
    
    %  add the Z series
    Xrhs = [Xrhs Z];
    %  construct basis functions
    XZbasis = Ord_Polynomial_N(Xrhs,D);
    % run regressions to fit data
    beta = Num_Stab_Approx(XZbasis,XYlhs,RM,penalty,1);
    beta = real(beta);
end

% construct valid beta matrix from betaguess
% find size of beta guess
[rbeta, cbeta] = size(beta);
% find number of terms in basis functions
[rbasis, ~] = size(Ord_Polynomial_N(ones(1,nx+nz),D)');
if rbeta > rbasis
    disp('rbeta > rbasis  truncating betaguess')
    beta = beta(1:rbasis,:);
    [rbeta, cbeta] = size(beta);
end
if cbeta > nx+ny
    disp('cbeta > cbasis  truncating betaguess')
    beta = beta(:,1:nx+ny);
    [rbeta, cbeta] = size(beta);
end
if rbeta < rbasis
    disp('rbeta < rbasis  adding extra zero rows to betaguess')
    beta = [beta; zeros(rbasis-rbeta,cbeta)];
    [rbeta, cbeta] = size(beta);
end
if cbeta < nx+ny
    disp('cbeta < cbasis  adding extra zero columns to betaguess')
    beta = [beta zeros(rbeta,nx+ny-cbeta)];
end
% find constants that yield theoretical SS
beta(1,:) = XYbar(1,1:nx)*(eye(nx)-beta(2:nx+1,1:nx));

% set starting values to SS if flag is on.
if X1 == -999
    X1 = XYbar(1:nx);
end

% set intial value of old coefficients
betaold = beta;

% begin iterations
dif_1d = 1;      
icount = 0;
[icount dif_1d 1]
while dif_1d > 1e-4*kdamp
    % update interation count
    icount = icount + 1;
    % stop if too many iterations have passed
    if icount > maxwhile
        break
    end
    
    % find convex combination of old and new coefficients
    beta = kdamp*beta + (1-kdamp)*betaold;
    betaold = beta;
    
    % find time series for XYap using approximate function
    % initialize series
    XYap = zeros(T,nx+ny);
    
    % find SS implied by current beta
    if fittype == 2
        Xbar = exp(beta(1,1:nx)*(eye(nx)-beta(2:1+nx,1:nx))^(-1));
    else
        Xbar = beta(1,1:nx)*(eye(nx)-beta(2:1+nx,1:nx))^(-1);
    end
    
    % % use theoretical SS vslues
    % Xbar = XYbar(:,1:nx);
    
    if ny > 0
        if fittype == 2
            Ybar = exp(ones(1,ny)*beta(1,1+nx:nx+ny) + log(Xbar)*beta(2:1+nx,1+nx:nx+ny));
        else
            Ybar = ones(1,ny)*beta(1,1+nx:nx+ny) + Xbar*beta(2:1+nx,1+nx:nx+ny);
        end
    else
        Ybar = [];
    end
    X1 = [Xbar Ybar];
    % find Xp & Y using approximate Xp & Y functions
    XYap(1,:) = GSSA_XYfunc(X1(1:nx),Z(1,:),beta,GSSAparams);
    for t=2:T
        XYap(t,:) = GSSA_XYfunc(XYap(t-1,1:nx),Z(t,:),beta,GSSAparams);
    end
    
    % generate XYex using the behavioral equations
    %  Judd, Mailar & Mailar call this y(t)
    [XYex,~] = GSSA_genex(XYap,Z,beta,J,epsi_nodes,weight_nodes,...
        GSSAparams,modelparams);

%     % watch the data converge using plots
%     for i=1:nx+ny
%         figure;
%         subplot(nx+ny,1,i)
%         plot([XYap(:,i) XYex(:,i)])
%     end
%     [XYap(1:10,:) XYex(1:10,:)]
%     [XYap(T-10:T,:) XYex(T-10:T,:)]
    
    % find new coefficient values
    % generate basis functions for Xap & Z
    %  get the X portion of XYap
    Xap = [X1(1:nx); XYap(1:T-1,1:nx)];
    if fittype == 2
        Xap = log(Xap);
        XYex = log(XYex);
    end
    %  add the Z series
    XZap = [Xap Z];
    %  construct basis functions
    XZbasis = Ord_Polynomial_N(XZap,D);
    % run regressions to fit data
    % Inputs:  "X" is a matrix of dependent variables in a regression, T-by-n,
    %          where n corresponds to the total number of coefficients in the 
    %          original regression (i.e. with unnormalized data);
    %          "Y" is a matrix of independent variables, T-by-N; 
    %          "RM" is the regression (approximation) method, RM=1,...,8:  
    %          1=OLS,          2=LS-SVD,    3=LAD-PP,    4=LAD-DP, 
    %          5=RLS-Tikhonov, 6=RLS-TSVD,  7=RLAD-PP,   8=RLAD-DP;
    %          "penalty"  is a parameter determining the value of the regulari-
    %          zation parameter for a regularization methods, RM=5,6,7,8;
    %          "normalize"  is the option of normalizing the data, 
    %          0=unnormalized data,  1=normalized data      
    beta = Num_Stab_Approx(XZbasis,XYex,RM,penalty,1);
    beta = real(beta);
    
    % evauate convergence criteria
    if icount == 1
        dif_1d = 1; 
        dif_beta = abs(1 - mean(mean(betaold./beta))); 
    else
        dif_1d =abs(1 - mean(mean(XYapold./XYap)));  
        dif_beta =abs(1 - mean(mean(betaold./beta)));  
        if isnan(dif_1d)
            dif_1d = 0;
            disp('There were problems with NaN for the convergence metric')
        end
        if isinf(dif_1d)
            dif_1d = 0;
            disp('There were problems with inf for the convergence metric')
        end
    end
    
    % replace old k values
    XYapold = XYap;
    
    % report results of iteration
    [icount dif_1d dif_beta]
end

[~,eulerr] = GSSA_genex(XYap,Z,beta,J,epsi_nodes,weight_nodes,...
    GSSAparams,modelparams);

end