% GSSA package
% version 1.4 written by Kerk L. Phillips  11/5/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables. 

function [nodes, weights] = GSSA_nodes(J,nz,SigZ)
% This function calculates nodes and weights for rectangular quadrature 
% used for taking expectations of behavioral eqns.  This setup uses equally
% spaced probabilities (weights).  This function is only suitable for the
% case of a single exogenous shock.
%
% This function takes 3 inputs:
% 1) J, the number of nodes used in the quadrature
% 2) nz, the integer number of elements in Z
% 3) SigZ, the variance of the shocks to Z
%
% The output is:
% 1) nodes, a vector of epsilon values
% 2) weights, a vector of probability weights

if nz < 2
    weights = ones(J,1)/J;
    cumprob = .5/J:1/J:1-.5/J;
    nodes = norminv(cumprob,0,SigZ);
else
    disp('too many elements in Z')
end

% check if the nodes look like a cummulative normal
% expnodes = weights*nodes'
% figure;
% plot(nodes)

end