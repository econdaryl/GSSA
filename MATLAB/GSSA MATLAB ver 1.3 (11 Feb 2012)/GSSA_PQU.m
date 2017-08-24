function [PP,QQ,UU] = GSSA_PQU(theta0);
global Zbar nx ny nz NN dyneqns

% Solve for the linear coefficient matrices PP & QQ for a linearization
% about the values specified in the vector "in", previously declared in
% earlier code somewhere.  This set of programs uses either the steady
% state where in = [Kbar; Kbar; Kbar; zbar; zbar] or the current state
% where in = [Kminus; Kminus; Kminus; Znow; Znow].

% rename the state about which the approximation will be made
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, TT] = RBCnumerderiv(dyneqns,theta0,nx,ny,nz);
Z0 = theta0(3*nx+2*ny+1)-Zbar;

% set variables and options for Uhlig's programs
[l_equ,m_states] = size(AA);
[l_equ,n_endog ] = size(CC);
[l_equ,k_exog  ] = size(DD);

% options for SOLVE.M & SOL_OUT.M
DO_QZ = 0;
message = '                                                                       ';
warnings = [];
DISPLAY_IMMEDIATELY = 0;
TOL = .000001; % Roots smaller than TOL are regarded as zero.
               % Complex numbers with distance less than TOL are regarded as equal.
if exist('MANUAL_ROOTS')~=1,
   MANUAL_ROOTS = 0; % = 1, if you want to choose your own
                               % roots, otherwise set = 0. 
                               % See SOLVE.M for instructions.
end;
if exist('IGNORE_VV_SING')~=1,
   IGNORE_VV_SING = 1; % =1: Ignores, if VV is singular.  
                       % Sometimes useful for sunspots.  Cross your fingers...  
end;
DISPLAY_ROOTS = 0;  % Set = 1, if you want to see the roots.

% run Uhlig's programs
solve;

UU = -(FF*PP+FF+GG)^(-1)*(TT+(FF*QQ+LL)*(NN*Z0-Z0));