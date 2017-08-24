function y = test4_defs(X, z, Xplus)
% caclulates the values of Y(t), r(t), w(t), C(t) & tau(t) for 
% the input values of X(t), X(t+1) & z(t)

global alpha lbar delta taubar d Bmax Bupp Blow Bmin
K = X(1);
k = X(2);
B = K - k;
Kplus = Xplus(1);
kplus = Xplus(2);
Bplus = Kplus - kplus;

Y = exp(z)*K^alpha*lbar^(1-alpha);
r = alpha*Y/K;
w = (1-alpha)*Y/lbar;
if B > Bmax
    tau = 0;
elseif B > Bupp
    tau = taubar*((Bmax-B)/(Bmax-Bupp));
elseif B > Blow
    tau = taubar;
elseif B > Bmin
    tau = 1 - (1-taubar)*((B-Bmin)/(Blow-Bmin));
else
    tau = 1;
end
c = (1-tau)*w*lbar + (1+r-delta)*k + d - kplus;

y = [Y; w; r; c; tau];