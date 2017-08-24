function out = test_dyn(in)
% Euler equations for the fixed labor Hansen model

global gam bet del

kpp = in(1);
kp =  in(2);
k =   in(3);
zp =  in(4);
z =   in(5);

[~, ~, cp, rp] = test_def(kp,zp,kpp);
[~, ~, c, ~] = test_def(k,z,kp);

out = bet*((c/cp)^gam)*(rp+1-del)-1;