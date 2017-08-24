function out = test2_dyn(in)
% Hansen model with variable labor
global gam bet del B mu

% nx = 2, ny = 0
kpp = in(1);
hpp = in(2);
kp =  in(3);
hp =  in(4);
k =   in(5);
h =   in(6);
zp =  in(7);
z =   in(8);

% % nx =1, ny = 1
% kpp = in(1);
% kp =  in(2);
% k =   in(3);
% hp =  in(4);
% h =   in(5);
% zp =  in(6);
% z =   in(7);

[yp, ip, cp, rp, wp] = test2_def(kp,hp,zp,kpp);
[y, i, c, r, w] = test2_def(k,h,z,kp);

out1 = (B*(1-h)^(-mu))/(c^(-gam)*w)-1;
out2 = bet*((c/cp)^gam)*(rp+1-del)-1;

out = [out1; out2];