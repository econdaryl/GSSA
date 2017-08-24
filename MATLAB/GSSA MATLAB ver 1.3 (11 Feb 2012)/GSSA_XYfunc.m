function XYp = GSSA_XYfunc(X,Z,beta)
% This is the approximation function that genrates Xp and Y from the
% current state.  inputs and outputs are row vectors
% Currently it allows for log-linear OLS or log-linear LAD forms.

% parameters
global D trunceqns
% options flags
global fittype dotrunc

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

% % truncate if needed depending on model
% if dotrunc == 1
%     XYp = trunceqns(XYp);
% end
% 
% end