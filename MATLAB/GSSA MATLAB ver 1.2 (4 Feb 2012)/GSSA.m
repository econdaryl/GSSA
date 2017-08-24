% GSSA function
% version 1.1 written by Kerk L. Phillips  2/5/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables.

% This function takes two inputs:
% 1) a guess for the steady state value of the endogneous state variables &
% jump variables, XYbarguess  It outputs the parameters for an
% approximation of the state transition function, X(t+1) = f(X(t),Z(t)) and
% the jump variable function Y(t) = g(X(t),Z(t)).
%    where X is a vector of nx endogenous state variables
%    Y is a vector of ny endogenous state variables
%    and Z is a vector of nz exogenous state variables
% 2) 3 initial guesses for the parameter matrices, AA, BB & CC
% 3) a string specifying the model's name, "modelname"

% This script requires two external functions:
%  1) One should be named "modelname"_dyn.m and should take as inputs:
%   X(t+2), X(t+1), X(t), Y(t+1), Y(t), Z(t+1) & Z(t)
%  It should output a with nx+ny elements column vector with the values of 
%  the model's behavioral equations (many of which will be Euler equations)
%  written in such a way that the equation evaluates to zero.  This 
%  function is used to find the steady state and in the GSSA algorithm 
%  itself.
%  2) A second function is named GSSA_fittype.m and is used to rehape the
%  vector of approximate function coefficients into matrices suitable for
%  simulating data.  (In this version these matrices are AA, BB, and CC.)

% The output is:
% 1) a vector of approximation coeffcients.
% 2) a vector of steady state values for X & Y
% 3) average Euler equation errors over a series of Monte Carlos.

% The GSSA_fittype.m function can be used to recover their matrix
% versions.

function [out,XYbarout,errors] = GSSA(XYbarguess,AAguess,BBguess,CCguess,modelname)

% data values
global X1 Z
% options flags
global fittype regconstant regtype numerSS quadtype
% parameters
global nx ny nz nc npar betar nobs XYpars XYbar NN SigZ dyneqns
global nodes weights
global AA BB CC beta

% create the name of the dynamic behavioral equations function
dyneqns = str2func([modelname '_dyn']);

% set numerical parameters (I include this in the model script)
% nx = 1;        %number of endogenous state variables
% ny = 0;        %number of jump variables
% nz = 1;        %number of exogenous state variables
%numerSS = 1;     %set to 1 if XYbargues is a guess, 0 if it is exact SS.

nobs = 20;     %number of observations in the simulation sample
ccrit = .001;    %convergence criterion for approximations
conv = .25;       %convexifier parameter for approximations
maxwhile = 500;  %maximimum iterations for approximations
nmc = 1;        %number of Monte Carlos for Euler error check
% fittype = 1;     %functional form for approximations
%                  %0 is linear (AVOID)
%                  %1 is log-linear
%                  %2 is quadratic (AVOID)
%                  %3 is log-quadratic
regconstant = 0; %choose 1 to include a constant in the regression fitting
regtype = 1;     %type of regression, 1 is LAD, 0 is OLS.
quadtype = 0;    %type of quadrature used
                 %0 is rectangular
J = 20;          %number of nodes for quadrature

nc = (nx+nz)*(nx+nz+1)/2;
npar = (nx*nx+nx*nz+nx+ny*nx+ny*nz+ny);
if fittype==2 || fittype==3
    npar = npar + nc;
end
if regconstant == 0;
    npar = npar - nx - ny;
end
npar
                 
% calculate nodes and weights for quadrature 
[nodes, weights] = GSSA_nodes(J,quadtype);

% set initial guess for coefficients
if regconstant == 1
    AA = [];
else
    if size(AAguess) ~= [1,nx+ny]
        AA = AAguess;
    else
        AA = zeros(1,nx+ny);
    end
end
if size(BBguess) ~= [nx+nz,nx+ny]
    BB = .1*ones(nx+nz,nx+ny);
else
    BB = BBguess;
end
if fittype==2 || fittype==3
    if size(CCguess) ~= [nc,nx+ny]
        CC = zeros(nc,nx+ny);
    else
        CC = CCguess;
    end
else
    CC = [];
end

beta = [AA; BB; CC];
[betar, betac] = size(beta);
nbeta = betar*betac
XYpars = reshape(beta,1,nbeta);

% find steady state numerically; skip if you already have exact values
if numerSS == 1
    options = optimset('Display','iter');
    XYbar = fsolve(@GSSA_ss,XYbarguess,options);
else
    XYbar = XYbarguess;
end

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
    XYparsold = XYpars;
    
    % find time series for XYap using approximate function
    XYap = GSSA_genap();
    
    % generate XYex using the behavioral equations
    %  Judd, Mailar & Mailar call this y(t)
    [XYex,~] = GSSA_genex();
    
    % find new coefficient values;
    XYpars = GSSA_fit(XYex,XYap,XYparsold);
    beta = reshape(XYpars,betar,nx+ny);
    [AA,BB,CC] = GSSA_fittype(beta);
    
    % evauate convergence criteria
    if icount == 1
        converge = 1;
    else
        converge = sum(sum(abs((XYap-XYapold)./XYap)),2)/(nobs*(nx+ny));  
        if isnan(converge)
            converge = 0;
            disp('There are problems with NaN for the convergence metric')
        end
        if isinf(converge)
            converge = 0;
            disp('There are problems with inf for the convergence metric')
        end
    end
    
    % replace old k values
    XYapold = XYap;
    
    % report results of iteration
    %[icount,converge]
    [icount converge XYpars]
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
global nx ny

T = size(Z,1);

% initialize series
XY = zeros(T,nx+ny);

% find Xp & Y using approximate Xp & Y functions
XY(1,:) = GSSA_XYfunc(X1,Z(1,:));
for t=2:T
    XY(t,:) = GSSA_XYfunc(XY(t-1,1:nx),Z(t,:));
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
function [Xp,Y] = GSSA_XYfunc(X,Z)
% This is the approximation function that genrates Xp and Y from the
% current state.  inputs and outputs are row vectors
% Currently it allows for log-linear OLS or log-linear LAD forms.

% parameters
global nx ny XYbar
global beta
% options flags
global fittype regconstant

Xbar = XYbar(1:nx);

% notation assumes data in column vectors, so individualn observations are
% row vectors.
% put appropriate linear & quadratic terms in indep list, we are removing
% the mean from the X's.
if fittype == 0  %linear
    indep = [(X-Xbar) Z];
elseif fittype == 1 %log-linear
    indep = [log(X./Xbar) Z];
elseif fittype == 2  %quaddratic
    sqterms = GSSA_sym2vec([(X-Xbar) Z]'*[(X-Xbar) Z]);
    indep = [(X-Xbar) Z sqterms];
elseif fittype == 3  %log quadratic
    sqterms = GSSA_sym2vec([log(X./Xbar) Z]'*[log(X./Xbar) Z]);
    indep = [log(X./Xbar) Z sqterms];
end

% add constants if needed
if regconstant == 1;
    indep = [ones(1,nx+ny) indep];
end

% create dependent variable
dep = indep*beta;

% convert to Xp's & Y's
if fittype==0 || fittype==2
    XYp = XYbar + dep;
elseif fittype==1 || fittype==3
    XYp = XYbar.*exp(dep);
end

% separate Xp's and Y's
Xp = XYp(1:nx);
if ny > 0
    Y = XYplus(nx+1:nx+ny);
else
    Y = [];
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
function XYparout = GSSA_fit(XYex,XYap,XYparguess)
% This function fits Xpex (Judd, Mailar & Mailar's y(t+1)) to Xpap (the X
% series from the approximation equations.

% data values
global X1 Z
% paramters
global nx nc XYbar
% options flags
global fittype regtype regconstant
% data to pass to GSSA_ADcalc function
global LADY LADX

[T,~] = size(XYex);

% independent variables
% get current period values, take deviations from SS, and add Z
Xbar = XYbar(1:nx);
Xap = [X1; XYap(1:T-1,1:nx)];
if fittype==0 || fittype==2 %levels
    Xap = Xap - repmat(Xbar,T,1);
elseif fittype==1 || fittype==3  %logarithms
    Xap = log(Xap) - repmat(log(Xbar),T,1);
end
if regconstant == 1
    LADX = [ones(T,1) Xap Z];
else
    LADX = [Xap Z];
end

% calculate and concatenate squared terms if needed
if fittype==2 || fittype==3 %quadratic
    sqterms = zeros(T,nc);
    for t=1:T
        sqterms(t,:) = GSSA_sym2vec([Xap(t,:) Z(t,:)]'*[Xap(t,:) Z(t,:)]);
    end
    LADX = [LADX sqterms];
end
        
% dependent variables
% take deviations
if fittype==0 || fittype==2  %levels
    LADY = XYex - repmat(XYbar,T,1);
elseif fittype==1 || fittype==3  %logarithms
    LADY = log(XYex) - repmat(log(XYbar),T,1);
end

% choose estimation method
if regtype == 1  %linear regression with LAD
    options = optimset('Display','on','MaxFunEvals',1000000,...
        'MaxIter',10000);
    XYparout = fminsearch(@GSSA_ADcalc,XYparguess,options);
elseif regtype == 0  %linear with OLSregression
    XYparout = LADY\LADX;
end

end

%%
function AD = GSSA_ADcalc(XYpars)
% Does LAD estimation

% data for AD estimation
global LADY LADX

% get number of regressors and regressands
[~,nX] = size(LADX);
[~,nY] = size(LADY);

beta = reshape(XYpars,nX,nY);
AD = sum(sum(abs(LADY - LADX*beta)),2);

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