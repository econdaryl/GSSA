function [y, i, c, r, w] = test2_def(k,h,z,kp)
% Hansen model with variable labor
global A del theta

y = A*(k^theta*(exp(z)*h)^(1-theta));
i = kp - (1-del)*k;
c = y - i;
r = theta*y/k;
w = (1-theta)*y/h;