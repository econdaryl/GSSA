D = 5;  % order of polynomial
T = 100;  % number of observations
nx = 1;
nz = 1;
NN = .9;

eps = randn(T,nz);
Z = zeros(T,nz);
Z(1,:) = eps(1,:);
for t=2:T
    Z(t,:) = Z(t-1,:)*NN + eps(t,:);
end
X = randn(T,nx);

XZbasis = Ord_Polynomial_N([X Z],D);
[~,nreg] = size(XZbasis);

beta = [rand(nx+1,nx)-.5; zeros(nreg-nx-1,nx)];

for t=2:T
    X(t,:) = Ord_Polynomial_N([X(t-1,:) Z(t-1,:)],D)*beta;
end

Xfinal = Ord_Polynomial_N([X(T,:) Z(T,:)],D)*beta;

Xp = [X(2:T,:); Xfinal]
XZbasis = Ord_Polynomial_N([X Z],D)

B = Num_Stab_Approx(XZbasis,Xp,2,7,1);

[beta B B - beta]