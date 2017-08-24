% GSSA package
% version 1.4 written by Kerk L. Phillips  11/5/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables.

function XYp = GSSA_XYfunc(X,Z,beta,GSSAparams)
global trunceqns
% This is the approximation function that genrates Xp and Y from the
% current state.  inputs and outputs are row vectors
% Currently it allows for log-linear OLS or log-linear LAD forms.
%
% This function takes 4 inputs:
% 1) X, the vector of current period endogenous state variables
% 2) Z, the vector of current period exogenous state variables
% 3) beta, the vector polynomial coefficients
% 4) GSSAparams, the vector of paramter values from the GSSA function
%
% The output is:
% XYp, a vector of the next period exogenous state variables and the 
% current period jump variables
%
% It requires the following subroutine written by by Kenneth L. Judd, 
% Lilia Maliar and Serguei Maliar which are available as a zip file from
% Lilia & Setguei Mailar's webapage at:
% http://www.stanford.edu/~maliars/Files/Codes.html
% "Ord_Polynomial_N.m" constructs the sets of basis functions for ordinary 
%                      polynomials of the degrees from one to five, for
%                      the N-country model

% read in GSSA parameters
dotrunc      = GSSAparams(9);
fittype      = GSSAparams(10);
D            = GSSAparams(13);

if fittype == 2 %log-linear
    XZ = [log(X) Z];
else
    XZ = [X Z];
end

% create dependent variable
% using basis functions of XZ (includes constants) * beta
XYp =  Ord_Polynomial_N(XZ,D)*beta;

% convert if needed
if fittype == 2
    XYp = exp(XYp);
end

% truncate if needed depending on model
if dotrunc == 1
    XYp = trunceqns(XYp);
end

end