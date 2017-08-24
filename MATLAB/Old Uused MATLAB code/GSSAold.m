% GSSA function
% version 1.3 written by Kerk L. Phillips  2/11/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables.

% It requires the following subroutines written by by Kenneth L. Judd, 
% Lilia Maliar and Serguei Maliar which are available as a zip file from
% Lilia & Setguei Mailar's webapage at:
% http://www.stanford.edu/~maliars/Files/Codes.html

% This function takes two inputs:
% 1) a guess for the steady state value of the endogneous state variables &
% jump variables, XYbarguess  It outputs the parameters for an
% approximation of the state transition function, X(t+1) = f(X(t),Z(t)) and
% the jump variable function Y(t) = g(X(t),Z(t)).
%    where X is a vector of nx endogenous state variables
%    Y is a vector of ny endogenous state variables
%    and Z is a vector of nz exogenous state variables
% 2) an initial guess for the parameter matrix, beta
% 3) a string specifying the model's name, modelname

% This script requires one external function:
%  It should be named "modelname"_dyn.m and should take as inputs:
%    X(t+2), X(t+1), X(t), Y(t+1), Y(t), Z(t+1) & Z(t)
%  It should output a with nx+ny elements column vector with the values of 
%  the model's behavioral equations (many of which will be Euler equations)
%  written in such a way that the equation evaluates to zero.  This 
%  function is used to find the steady state and in the GSSA algorithm 
%  itself.

% The output is:
% 1) a vector of approximation coeffcients, out.
% 2) a vector of steady state values for X & Y, XYbarout.
% 3) average Euler equation errors over a series of Monte Carlos, errors

function [out,XYbarout,errors] = GSSA(XYbarguess,beta,modelname)

% data values
global X1 Z
% options flags
global fittype regtype numerSS quadtype
% parameters
global nx ny nz nc npar betar nbeta nobs XYpars XYbar NN SigZ dyneqns
global nodes weights
global AA BB CC beta
% model
global model

% create the name of the dynamic behavioral equations function
dyneqns = str2func([modelname '_dyn']);

% set numerical parameters (I include this in the model script)
% nx = 1;        %number of endogenous state variables
% ny = 0;        %number of jump variables
% nz = 1;        %number of exogenous state variables
%numerSS = 1;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.

model = modelname;
nobs = 1000;      %number of observations in the simulation sample
ccrit = 10^-10;    %convergence criterion for approximations
conv = .025;       %convexifier parameter for approximations (weight on new)
maxwhile = 1000;  %maximimum iterations for approximations
nmc = 10;         %number of Monte Carlos for Euler error check
fittype = 2;     %functional form for approximations
                 %0 is linear
                 %1 is log-linear
                 %2 is quadratic
                 %3 is log-quadratic
regtype = 3;     %type of regression
                 %0 is OLS
                 %1 is LAD
                 %2 is SVD
                 %3 is truncated SVD
quadtype = 0;    %type of quadrature used
                 %0 is rectangular
J = 20;          %number of nodes for quadrature

na = 1;
nb = (nx+nz);
nc = (nx+nz)*(nx+nz+1)/2;
npar = na+nb;
if fittype==2 || fittype==3
    npar = npar + nc;
end
npar = (nx+ny)*npar
                 
% calculate nodes and weights for quadrature 
[nodes, weights] = GSSA_nodes(J,quadtype);

% find steady state numerically; skip if you already have exact values
if numerSS == 1
    options = optimset('Display','iter');
    XYbar = fsolve(@GSSA_ss,XYbarguess,options)
else
    XYbar = XYbarguess;
end

% set initial guess for coefficients
AA = .1*ones(1,nx+ny);

if sum(abs(size(BBguess)-[nx+nz,nx+ny]),2) ~= 0
    BB = zeros(nx+nz,nx+ny);
else
    BB = BBguess;
end

if fittype==2 || fittype==3
    if sum(abs(size(CCguess)-[nc,nx+ny]),2) ~= 0
        CC = zeros(nc,nx+ny);
    else
        CC = CCguess;
    end
else
    CC = [];
end

beta = [AA; BB; CC]
[betar, betac] = size(beta);
nbeta = betar*betac
XYpars = reshape(beta,1,nbeta);

% start at SS
X1 = XYbar(1:nx);

% generate a random history of Z's to be used throughout
Z = zeros(nobs,nz);
eps = randn(nobs,nz)*SigZ;
for t=2:nobs
    Z(t,:) = Z(t-1,:)*NN + eps(t,:);
end

% set intial value of old coefficients
XYparsold = XYpars;
betaold = beta;

% begin iterations
converge = 1;
icount = 0;
[icount converge XYpars]
while converge > ccrit
    % update interation count
    icount = icount + 1;
    % stop if too many iterations have passed
    if icount > maxwhile
        break
    end
    
    % find convex combination of old and new coefficients
    XYpars = conv*XYpars + (1-conv)*XYparsold;
    beta = conv*beta + (1-conv)*betaold;
    XYparsold = XYpars;
    betaold = beta;
    
    % find time series for XYap using approximate function
    XYap = GSSA_genap();
    
    % generate XYex using the behavioral equations
    %  Judd, Mailar & Mailar call this y(t)
    [XYex,~] = GSSA_genex();
    
%     for i=1:nx+ny
%         figure;
%         subplot(nx+ny,1,i)
%         plot([XYap(:,i) XYex(:,i)])
%     end
%     [XYap(1:10,:) XYex(1:10,:)]
    
    % find new coefficient values;
    beta = GSSA_fit(XYex,XYap);
    
    XYpars = reshape(beta,1,nbeta);
    [AA,BB,CC] = GSSA_fittype(beta);
    
    % evauate convergence criteria
    if icount == 1
        converge = 1;
    else
        converge = sum(sum(abs((XYap-XYapold)./XYap)),2)/(nobs*(nx+ny));  
        if isnan(converge)
            converge = .5;
            disp('There are problems with NaN for the convergence metric')
        end
        if isinf(converge)
            converge = .5;
            disp('There are problems with inf for the convergence metric')
        end
    end
    
    % replace old k values
    XYapold = XYap;
    
    % report results of iteration
    %[icount,converge]
    [icount converge XYpars]
    beta
end

out = XYpars;
XYbarout = XYbar;

% run Monte Carlos to get average Euler errors
errors = 0;
for m=1:nmc
    % create a new Z series
    Z = zeros(nobs,nz);
    eps = randn(nobs,nz)*SigZ;
    for t=2:nobs
        Z(t,:) = Z(t-1,:)*NN + eps(t,:);
    end
    
    % generate data & calcuate the errors, add this to running average
    [~,temp] = GSSA_genex();
    errors = errors*(m-1)/m + temp/m;
    m
end

end

%%
function XY = GSSA_genap()
% This function generates approximate values of Xp using the approximation
% equations in GSSA_XYfunc.m

% data values
global Z X1
% parameters
global nx ny beta
% model
global model

T = size(Z,1);

% initialize series
XY = zeros(T,nx+ny);

% find SS implied by current beta
Xbar = beta(1,1:nx)*(eye(nx)-beta(2:1+nx,1:nx))^(-1);
if ny > 0
    Ybar = ones(1,ny)*beta(1,1+nx:nx+ny) + Xbar*beta(2:1+nx,1+nx:nx+ny);
else
    Ybar = [];
end
X1 = [Xbar Ybar];

% find Xp & Y using approximate Xp & Y functions
XY(1,:) = GSSA_XYfunc(X1,Z(1,:));
for t=2:T
    XY(t,:) = GSSA_XYfunc(XY(t-1,1:nx),Z(t,:));
    % model specific truncations
%     if model == 'test2'
%         % h is between zero and one
%         if XY(t,2) > 1-10^-10
%             XY(t,2) = 1-10^-10;
%         elseif XY(t,2) < 10^-10
%             XY(t,2) = 10^-10;
%         end
%         % k is positive
%         if XY(t,1) < 10^-10
%             XY(t,1) = 10^-10;
%         end
%     end
end

end

%%
function [XYex,eulerr] = GSSA_genex()
% This function generates values of Xp using the behavioral equations
% in "name"_dyn.n.  This is y(t) from Judd, Mailar & Mailar.

% data values
global Z X1
% options flags
global quadtype
% parameters
global nx ny nz NN nobs dyneqns nodes weights

T = nobs;
[~,J] =size(nodes);

% initialize series
XYex = zeros(T,nx+ny);

% find Xp & Y using approximate Xp & Y functions
XY = GSSA_genap();
Xp = XY(:,1:nx);
Y = XY(:,nx+1:nx+ny);
eulert = zeros(T,1);

% find X
X = [X1; Xp(1:T-1,:)];

% find EZp & EXpp using law of motion and approximate X & Y functions
if quadtype == 0 && nz < 2
    for t=1:T
        EZpj = zeros(J,nz);
        EXYj = zeros(J,nx+ny);
        EXppj = zeros(J,nx);
        if ny > 0
            EYpj = zeros(J,ny);
        end
        EGFj = zeros(J,nx+ny);
        XYexj = zeros(J,nx+ny);
        % integrate over discrete intervals
        for j=1:J
            % find Ezp using law of motion
            EZpj(j,:) = Z(t,:)*NN + nodes(j);

            % find EXpp & EYp using approximate functions
            EXYj(j,:) = GSSA_XYfunc(Xp(t,:),EZpj(j,:));
            EXppj(j,:) = EXYj(j,1:nx);
            if ny > 0
               EYpj(j,:) = EXYj(j,nx+1:nx+ny);
            end

            % find expected G & F values using nonlinear behavioral 
            % equations since GSSA-dyn evaluates to zero at the fixed point
            % and it needs to evaluate to one, add ones to each element.
            if ny > 0
              EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
                          EYpj(j,:)';Y(t,:)';EZpj(j,:)';Z(t,:)]')' + 1;
            else
              EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
                          EZpj(j,:)';Z(t,:)]')' + 1; 
            end

            % find Judd, Mailar & Mailar's y
            XYexj(j,:) = EGFj(j,:).*[Y(t,:) Xp(t,:)];  
        end

        % sum over J
        XYex(t,:) = weights*XYexj;
        eulert(t,:) = weights*(EGFj-1)*ones(nx+ny,1)/(nx+ny);
    end
end
eulerr = ones(1,T)*eulert/T;

end


%%
function XYp = GSSA_XYfunc(X,Z)
% This is the approximation function that genrates Xp and Y from the
% current state.  inputs and outputs are row vectors
% Currently it allows for log-linear OLS or log-linear LAD forms.

% parameters
global nx nz
global beta
% options flags
global fittype

% notation assumes data in column vectors, so individualn observations are
% row vectors.
% put appropriate linear & quadratic terms in indep list, we are removing
% the mean from the X's.
if fittype == 0  %linear
    indep = [X Z];
elseif fittype == 1 %log-linear
    indep = [log(X) Z];
elseif fittype == 2  %quaddratic
    sqterms = GSSA_sym2vec([X Z]'*[X Z]);
    indep = [X Z sqterms];
elseif fittype == 3  %log quadratic
    sqterms = GSSA_sym2vec([log(X) Z]'*[log(X) Z]);
    indep = [log(X) Z sqterms];
end
indep = [1 indep];

% create dependent variable
dep = indep*beta;

% convert to Xp's & Y's
if fittype==0 || fittype==2
    XYp = dep;
elseif fittype==1 || fittype==3
    XYp = exp(dep);
end

end

%%
function out = GSSA_ss(XYbar)
% This function finds the steady state using numerical methods

% parameters
global nx ny nz dyneqns

Xbar = XYbar(1:nx);
Ybar = XYbar(nx+1:nx+ny);
out = dyneqns([Xbar'; Xbar'; Xbar'; Ybar'; Ybar'; zeros(nz,1); zeros(nz,1)]);

end

%%
function betaout = GSSA_fit(XYex,XYap)
% This function fits Xpex (Judd, Mailar & Mailar's y(t+1)) to Xpap (the X
% series from the approximation equations.

% data values
global X1 Z
% paramters
global nx nz nc nbeta npar XYbar beta
% options flags
global fittype regtype
% data to pass to GSSA_ADcalc function
global Yfit Xfit

[T,~] = size(XYex);

% parameter vector
XYparguess = reshape(beta,1,nbeta);

% normalize variables
% independent Xfit
Xfit = [X1; XYap(1:T-1,1:nx)];
if fittype==1 || fittype==3
    Xfit = log(Xfit);
end
Xfit = [Xfit Z];
% calculate and concatenate squared terms if needed
if fittype==2 || fittype==3 %quadratic
    sqterms = zeros(T,nc);
    for t=1:T
        sqterms(t,:) = GSSA_sym2vec(Xfit(t,:)'*Xfit(t,:));
    end
    Xfit = [Xfit sqterms];
end
Xmu = mean(Xfit);
Xsig = std(Xfit);
Xfit = (Xfit - repmat(Xmu,T,1))./repmat(Xsig,T,1);

% dependent Yfit
Yfit = XYex;
if fittype==1 || fittype==3
    Yfit = log(Yfit);
end
Ymu = mean(Yfit);
Ysig = std(Yfit);
Yfit = (Yfit - repmat(Ymu,T,1))./repmat(Ysig,T,1);

% % plot data
% figure;
% scatter(Yfit(:,1),Xfit(:,1),5,'filled')

% choose estimation method and find new parameters
if regtype == 0  %linear with OLSregression
    betaout = Yfit\Xfit
elseif regtype == 1  %linear regression with LAD
    options = optimset('Display','on','MaxFunEvals',1000000,...
        'MaxIter',10000);
    XYparout = fminsearch(@GSSA_ADcalc,XYparguess,options);
    betaout = reshape(XYparout,nX,nY);
elseif regtype == 2  %SVD estimate
    [U,S,V] = svd(Xfit,0);
    betaout = V*S^(-1)*U'*Yfit;
elseif regtype == 3  %truncated SVD estimate
    % set truncation criterion
    kappa = 10^14;
    % do SVD decomposition
    [U,S,V] = svd(Xfit,0);
    % find ill-conditioned components and remove
    sumS = sum(S);
    sumS = sumS(1)./sumS;
    [~,cols] = size(S);
    r = 1;
    i = 1;
    while i<cols+1
        if sumS(i)<kappa
            r = i;
        else
           break
        end
        i = i+1;
    end
    S = S(1:r,1:r);
    U = U(:,1:r);
    V = V(:,1:r);
    % get estimate
    betaout = V*S^(-1)*U'*Yfit;
end
% adjust for normalization of means and variances
[~,ncol] = size(Xsig);
betaout = (ones(1,ncol)./Xsig)'*Ysig.*betaout;
beta0 = Ymu - Xmu*betaout;
betaout = [beta0; betaout];

end

%%
function AD = GSSA_ADcalc(XYpars)
% Does LAD estimation

% data for AD estimation
global Yfit Xfit

% get number of regressors and regressands
[~,nX] = size(Xfit);
[~,nY] = size(Yfit);

beta = reshape(XYpars,nX,nY);
AD = sum(sum(abs(Yfit - Xfit*beta)),2);

end

%%
function [nodes, weights] = GSSA_nodes(J,quadtype)
% does quadrature on expectations

% parameters
global nz SigZ
% calculate nodes and weights for rectangular quadrature used for taking 
% expectations of behavioral eqns.  This setup uses equally spaced 
% probabilities (weights)

if quadtype == 0 && nz < 2
    weights = ones(1,J)/J;
    cumprob = .5/J:1/J:1-.5/J;
    nodes = norminv(cumprob,0,SigZ);
end
% if isGpuAvailable == 1
%     nodes = gpuArry(nodes);
% end

% check if the nodes look like a cummulative normal
% expnodes = weights*nodes'
% figure;
% plot(nodes)

end


%%
function A = GSSA_sym2vec(B)
% B is a symmetric  matrix
% A is a row vectorization of it's upper triangular portion
A = [];
for k = 1:size(B,1)
    A = [A B(k,1:k)];
end

end