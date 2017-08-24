function [AA,BB,CC] = GSSA_fittype(beta)
global regconstant fittype nx nz nc
% convert vector of parameters into matrices for a quadratic model
% [X(t+1) Y(t)] = AA + BB * [X(t) Z(t)] + CC * symvect{[X(t) Z(t)]'*[X(t) Z(t)]}

if fittype==0 || fittype==1
    if regconstant==1
        AA = beta(1,:);
        BB = beta(1+1:1+nx+nz,:);
        CC = [];
    else
        AA = [];
        BB = beta(1:nx+nz,:);
        CC = [];
    end    
elseif fittype==2 || fittype==3
    if regconstant==1
        AA = beta(1,:);
        BB = beta(1+1:1+nx+nz,:);
        CC = beta(1+nx+nz+1:1+nx+nz+nc,:);
    else
        AA = [];
        BB = beta(1:nx+nz,:);
        CC = beta(nx+nz+1:nx+nz+nc,:);
    end
end

end