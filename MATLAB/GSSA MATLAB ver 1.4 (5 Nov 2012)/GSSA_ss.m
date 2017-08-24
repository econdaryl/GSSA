% GSSA package
% version 1.4 written by Kerk L. Phillips  11/5/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables.

function out = GSSA_ss(XYbar)
global nx ny nz dyneqns
% GSSA.m uses this function along with the fsolve command to find the 
% steady state using numerical methods.
%
% This function takes as an input:
% XYbar, a vector of steady state values for X and Y
%
% The output is:
% out, deviations of the model characterizing equations, which should be
% zero at the steady state

Xbar = XYbar(1:nx);
Ybar = XYbar(nx+1:nx+ny);
out = dyneqns([Xbar'; Xbar'; Xbar'; Ybar'; Ybar'; ...
              zeros(nz,1); zeros(nz,1)]);

end