function A = GSSA_sym2vec(B)
% B is a symmetric  matrix
% A is a row vectorization of it's upper triangular portion
A = [];
for k = 1:size(B,1)
    A = [A B(k,1:k)];
end

end

