function out = test_dyn(in, params)
% Euler equations for the fixed labor Hansen model

A      = params(1);
theta  = params(2);
del    = params(3);
bet    = params(4);
hbar   = params(5);
gam    = params(6);
rho    = params(7);
sig    = params(8);

kpp = in(1);
kp =  in(2);
k =   in(3);
zp =  in(4);
z =   in(5);

[~, ~, cp, rp] = test_def(kp,zp,kpp,params);
[~, ~, c, ~] = test_def(k,z,kp,params);

out = bet*((c/cp)^gam)*(rp+1-del)-1;