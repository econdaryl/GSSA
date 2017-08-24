function e = test4_dyn(x)
% caclulates the deviation of the Euler equation from zero for input values
% of K(t-1), K(t), K(T+1), z(t) & z(t+1)
% note we want this to be zero in the steady state (not one) so that we can
% use fsolve to find SS values, if we wish.
global bet delta gamma tau lbar d

% read in values
Kplus = x(1);
kplus = x(2);
Know   = x(3);
know   = x(4);
Kminus  = x(5);
kminus  = x(6);
zplus   = x(7);
znow  = x(8);

Bminus = Kminus - kminus;
Bnow = Know - know;
Bplus = Kplus - kplus;

Xminus = [Kminus kminus];
Xnow = [Know know];
Xplus = [Kplus kplus];

% get current period definitions
temp = test4_defs(Xminus, znow, Xnow);
Ynow = temp(1);
wnow = temp(2);
rnow = temp(3);
cnow = temp(4);
taunow = temp(5);

% get next period definitions
temp = test4_defs(Xnow, zplus, Xplus);
Yplus = temp(1);
wplus = temp(2);
rplus = temp(3);
cplus = temp(4);
tauplus = temp(5);

% calculate Euler equations
e1 = 1-bet*(cnow/cplus)^(gamma)*(1+rplus-delta);
e2 = taunow*wnow*lbar + Bminus*(1+rnow-delta) - d - Bnow;
e = [e1; e2];