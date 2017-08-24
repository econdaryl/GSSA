function out = test3_dyn(in)
% 3-period OLG model
global gam bet del

% disp('test3_dyn')

k1p = in(1);
k2p = in(2);
k1 =  in(3);
k2 =  in(4);
k1m = in(5);
k2m = in(6);
zp =  in(7);
z =   in(8);

[~, ~, ~, ~, c2p, c3p, rp, ~] = test3_def(k1,k2,zp,k1p,k2p);
[~, ~, ~, c1, c2, ~, ~, ~] = test3_def(k1m,k2m,z,k1,k2);

out1 = bet*((c1/c2p)^gam)*(rp+1-del)-1;   %determines k1p
out2 = bet*((c2/c3p)^gam)*(rp+1-del)-1;   %determines k2p

out = [out1; out2];