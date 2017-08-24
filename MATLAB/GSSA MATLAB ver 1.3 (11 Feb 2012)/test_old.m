% Hansen's model with a CES production function
clear
global A gam bet del theta rho sig hbar
global nx ny nz nobs NN

%set model parameters
A = 1;
theta = .33;
del = .025;
bet = .995;
hbar = .3;
gam = 2;
rho = .9;
sig = .02;
%set numerical parameters
nx = 1;
ny = 0;
nz = 1;
options = optimset('Display','iter','TolFun',.000000000001,'TolX',.000000000001);
numerical = 0; %choose 1 to use numerical solutions for the simulation
nobs = 251;
nmc = 1;

zbar = 0;
%find SS numerically
kbarN = fsolve(@test_ss,10,options);
[ybarN, ibarN, cbarN, rbarN, wbarN] = HW05adefs(kbarN,zbar,kbarN);
barN = [kbarN; ybarN; ibarN; cbarN; rbarN; wbarN]

%find SS algebraically
kbarA = hbar*((theta*A)/(1/bet+del-1))^(1/(1-theta));
[ybarA, ibarA, cbarA, rbarA, wbarA] = HW05adefs(kbarA,zbar,kbarA);
barA = [kbarA; ybarA; ibarA; cbarA; rbarA; wbarA]

%calculate and display percent differences between two methods
diff = log(barA./barN)

NN = rho;
%find derivatives and coefficients numerically
in = [kbarN; kbarN; kbarN; 0; 0];
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM] = RBCnumerderiv(@HW05adyn,in,nx,ny,nz);

root1 = (-GG+sqrt(GG^2-4*FF*HH))/(2*FF);
root2 = (-GG-sqrt(GG^2-4*FF*HH))/(2*FF);
if root1 < 1
    PPN = root1;
elseif root2 < 1
    PPN = root2;
else
    disp('no stable numerical roots')
end
QQN = -(LL*NN+MM)/(FF*NN+FF*PPN+GG);
coeffsN = [FF GG HH LL MM PPN QQN]

%find derivatives analytically
FF = gam*kbarA/cbarA;
GG = gam*(bet*rbarA*(theta-1)-theta*ybarA/cbarA-(2-del)*kbarA/cbarA);
HH = gam*((1-del)*kbarA/cbarA+theta*ybarA/cbarA);
LL = gam*((bet*rbarA-ybarA/cbarA)*(1-theta));
MM = gam*ybarA/cbarA*(1-theta);

root1 = (-GG+sqrt(GG^2-4*FF*HH))/(2*FF);
root2 = (-GG-sqrt(GG^2-4*FF*HH))/(2*FF);
if root1 < 1
    PPA = root1;
elseif root2 < 1
    PPA = root2;
else
    disp('no stable analytical roots')
end
QQA = -(LL*NN+MM)/(FF*NN+FF*PPA+GG);
coeffsA = [FF GG HH LL MM PPA QQA]

%calculate and display percent differences between two methods
diff = log(coeffsA./coeffsN)

% choose PP & QQ to use for simulations
if numerical == 1
    PP=PPN;
    QQ=QQN;
else
    PP=PPA;
    QQ=QQA;
end

mcmoments = zeros(8,6);
for m = 1:nmc
    %initialize series for state variables
    Xtilde = zeros(nobs,1);
    z = zeros(nobs,1);

    %generate eps shock series
    eps = sig*randn(nobs,1);

    %iteratively generate time series
    Xtilde(1) = 0;  %start at SS
    z(1) = 0;
    for t=2:nobs
        z(t) = rho*z(t-1) + eps(t);
        Xtilde(t) = PP*Xtilde(t-1) + QQ*z(t);
    end
    
    %transform Xtilde into k
    k = kbarA*exp(Xtilde);
    
    % get all other time series
    y = zeros(nobs,1);
    i = zeros(nobs,1);
    c = zeros(nobs,1);
    r = zeros(nobs,1);
    w = zeros(nobs,1);
    for t=2:nobs;
        [y(t), i(t), c(t), r(t), w(t)] = HW05adefs(k(t-1),z(t),k(t));
    end
    u = (c.^(1-gam)-ones(nobs,1))./(1-gam);
    
    % appropriately lag X to get k timing correct
    k = k(1:nobs-1);
    
    %put all time sereis into a matrix
    data = [y i c r w A*exp(z) u];
    % drop first observation
    data = data(2:nobs,:);
    % add k
    data = [k data];
    
    %calculate moments
    moments = zeros(8,6);
    means = mean(data);
    stdevs = std(data);
    coefvars = stdevs./means;
    correls = corrcoef(data);
    data2 = [data(1:nobs-2,:) data(2:nobs-1,:)];
    autocorrs = corrcoef(data2);
    for i=1:8
        moments(i,:) = [means(i) stdevs(i) coefvars(i) correls(i,2) correls(i,7) autocorrs(i,i+8)];
    end
    mcmoments = mcmoments*(m-1)/m + moments/m;
end
mcmoments

figure;
for i=1:8
    subplot(4,2,i)
    plot(data(:,i),'k-');
end