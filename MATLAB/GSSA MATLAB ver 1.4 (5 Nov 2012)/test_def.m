function [y, i, c, r, w] = test_def(k,z,kp,params)
% useful definitions for the Hansen model

A      = params(1);
theta  = params(2);
del    = params(3);
bet    = params(4);
hbar   = params(5);
gam    = params(6);
rho    = params(7);
sig    = params(8);

y = A*(k^theta*(exp(z)*hbar)^(1-theta));
i = kp - (1-del)*k;
c = y - i;
r = theta*y/k;
w = (1-theta)*y/hbar;