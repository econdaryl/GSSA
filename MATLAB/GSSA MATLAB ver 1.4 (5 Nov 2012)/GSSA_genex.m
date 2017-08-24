% GSSA package
% version 1.4 written by Kerk L. Phillips  11/5/2012
% Implements a version of Judd, Mailar & Mailar's (Quatitative Economics,
% 2011) Generalized Stochastic Simulation Algorithm with jump variables. 

function [XYex,eulerr] = GSSA_genex(XY,Z,beta,J,epsi_nodes,weight_nodes,...
    GSSAparams,modelparams)
global dyneqns
% This function generates values of Xp using the behavioral equations
% in "name"_dyn.n.  This is y(t) from Judd, Mailar & Mailar.
%
% This function takes 8 inputs:
% 1) XY, a t-by-(nx+ny) matrix of values generated using the fitted
%    functions
% 2) Z, a t-by-nz matrix of exogenous variables
% 3) beta, a maxtrix of polynomial coefficients for the fitted functions
% 4) J, the number of nodes used in the quadrature
% 5) epsi_nodes, a vector of epsilon values
% 6) weight_nodes, a vector of probability weights
% 7) GSSAparams, the vector of paramter values from the GSSA function
% 8) modelparams, a vector of model specific parameter values passed to the
%    model dynamic function named dyneqns
%
% The output is:
% 1) XYex, the matrix of expected values for X and Y
% 2) eulerr, the aveage value of the Euler error over the sample

% read in GSSA parameters
nx           = GSSAparams(1);
ny           = GSSAparams(2);
nz           = GSSAparams(3);
NN           = GSSAparams(17);
X1       = GSSAparams(19);
[T,~] = size(XY);

% initialize series
XYex = zeros(T,nx+ny);

% find X, Xp & Y using approximate parts of XY
Xp = XY(:,1:nx);
X = [X1(1:nx); Xp(1:T-1,:)];
Y = XY(:,nx+1:nx+ny);
eulert = zeros(T,1);

% find EZp & EXpp using law of motion and approximate X & Y functions
    for t=1:T
        EZpj = zeros(J,nz);
        EXYj = zeros(J,nx+ny);
        EXppj = zeros(J,nx);
        if ny > 0
            EYpj = zeros(J,ny);
        end
        EGFj = zeros(J,nx+ny);
        XYexj = zeros(J,nx+ny);
        % integrate over discrete intervals
        for j=1:J
            % find Ezp using law of motion
            EZpj(j,:) = Z(t,:)*NN + epsi_nodes(j);

            % find EXpp & EYp using approximate functions
            EXYj(j,:) = GSSA_XYfunc(Xp(t,:),EZpj(j,:),beta,GSSAparams);
            EXppj(j,:) = EXYj(j,1:nx);
            if ny > 0
               EYpj(j,:) = EXYj(j,nx+1:nx+ny);
            end
            
            % find expected G & F values using nonlinear behavioral 
            % equations since GSSA-dyn evaluates to zero at the fixed point
            % and it needs to evaluate to one, add ones to each element.
            if ny > 0
              EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
                          EYpj(j,:)';Y(t,:)';EZpj(j,:)';Z(t,:)]', ...
                          modelparams)' + 1;
%               % additive
%               EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
%                           EYpj(j,:)';Y(t,:)';EZpj(j,:)';Z(t,:)]')';
            else
              EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
                          EZpj(j,:)';Z(t,:)]', modelparams)' + 1;
%               % additive
%               EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
%                           EZpj(j,:)';Z(t,:)]')';
        
            end

            % find Judd, Mailar & Mailar's y
            XYexj(j,:) = EGFj(j,:).*[Y(t,:) Xp(t,:)];  
%             % additive
%             XYexj(j,:) = EGFj(j,:)+[Y(t,:) Xp(t,:)];  
        end

        % sum over J
        XYex(t,:) = weight_nodes'*XYexj;
        eulert(t,:) = weight_nodes'*(EGFj-1)*ones(nx+ny,1)/(nx+ny);
    end
eulerr = ones(1,T)*eulert/T;

end