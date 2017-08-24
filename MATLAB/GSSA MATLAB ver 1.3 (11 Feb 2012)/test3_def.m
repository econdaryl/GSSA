function [k, y, i, c1, c2, c3, r, w] = test3_def(k1,k2,z,k1p,k2p)
% 3-period OLG model
global A del theta

k = k1+k2;
kp = k1p + k2p;
y = A*(k^theta*(exp(z)*2)^(1-theta));
r = theta*y/k;
w = (1-theta)*y/2;
i = kp - (1-del)*k;
c1 = w - k1p;
c2 = w + (1+r)*k1 - k2p;
c3 = (1+r)*k2;