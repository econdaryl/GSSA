function y = test5_defs(K, z, Kplus)
% model with discrete marginal tax rates
% caclulates the values of Y(t), r(t), w(t), C(t), tau(t) and tax(t) for 
% the input values of K(t), K(t+1) & z(t)

global alpha lbar delta tau1 tau2 taxthreshold

Y = exp(z)*K^alpha*lbar^(1-alpha);
r = alpha*Y/K;
w = (1-alpha)*Y/lbar;
if w*lbar+(r-delta)*K<taxthreshold
    tax = tau1*(w*lbar+(r-delta)*K);
    tau = tau1;
else
    tax = tau1+taxthreshold+tau2*(w*lbar+(r-delta)*K-taxthreshold);
    tau = tau2;
end
% since d = tax;
c = w*lbar + (1+r-delta)*K - Kplus;

y = [Y; w; r; c; tau; tax];