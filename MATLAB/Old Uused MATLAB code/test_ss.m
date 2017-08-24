function out = test_ss(in)
kbar = in(1);
vector = [kbar; kbar; kbar; 0; 0];
out = test_dyn(vector);