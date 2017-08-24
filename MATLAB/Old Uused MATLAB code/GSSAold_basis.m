function Xbasis = GSSA_basis(X,D,PF)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% get sample size
[T,n] = size(X);
Xmu = mean(X);
Xsig = std(X);

%normalize data by romoving means and dividing by standard deviations
%Xnorm = (X-repmat(Xmu,T,1))./repmat(Xsig,T,1);

if PF == 0  %normal polynomials
    Xnorm = X;
    X2 = [];
    Xbasis = [];
    for t=1:T
        Xcum = [1 Xnorm(t,:)];
        
        if D > 1
            X2 = GSSA_sym2vec(Xnorm(t,:)'*Xnorm(t,:));
            Xcum = [Xcum X2];
        end
        
        if D > 2
            X3big = Xnorm(t,:)'*X2;
            F = 1; L = 1;
            X3big(1,1);
            X3 = [];
            for i=2:n
                F = L + 1; L = F + i -1;
                %[F L i n]
                X3 = [X3 GSSA_sym2vec(X3big(1:i,F:L))];
            end
            Xcum = [Xcum X3];
        end
        Xbasis = [Xbasis; Xcum];
    end
end
rank(Xbasis)

end