function [y, i, c, r, w] = test_def(k,z,kp)
% useful definitions for the Hansen model
global A del theta hbar

y = A*(k^theta*(exp(z)*hbar)^(1-theta));
i = kp - (1-del)*k;
c = y - i;
r = theta*y/k;
w = (1-theta)*y/hbar;