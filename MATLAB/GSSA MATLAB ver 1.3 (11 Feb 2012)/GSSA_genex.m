function [XYex,eulerr] = GSSA_genex(XY,beta)
% This function generates values of Xp using the behavioral equations
% in "name"_dyn.n.  This is y(t) from Judd, Mailar & Mailar.

% data values
global X1 Z
% parameters
global nx ny nz NN dyneqns
global J epsi_nodes weight_nodes

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
            EXYj(j,:) = GSSA_XYfunc(Xp(t,:),EZpj(j,:),beta);
            EXppj(j,:) = EXYj(j,1:nx);
            if ny > 0
               EYpj(j,:) = EXYj(j,nx+1:nx+ny);
            end
            
            % find expected G & F values using nonlinear behavioral 
            % equations since GSSA-dyn evaluates to zero at the fixed point
            % and it needs to evaluate to one, add ones to each element.
            if ny > 0
              EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...?
                          EYpj(j,:)';Y(t,:)';EZpj(j,:)';Z(t,:)]')' + 1;
%               % additive
%               EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
%                           EYpj(j,:)';Y(t,:)';EZpj(j,:)';Z(t,:)]')';
            else
              EGFj(j,:) = dyneqns([EXppj(j,:)';Xp(t,:)';X(t,:)';...
                          EZpj(j,:)';Z(t,:)]')' + 1;
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