function XYtrunc = test2_trunc(XY)
% XY is an nx+ny by 1 vector

% XY(1) is capital
% XY(2) is labor hours

eps = .000001;
% capital cannot be negative
if XY(1) < eps
    XY(1) = eps;
end

% hours must lie between 0 and 1
eps = .1;
if XY(2) < eps
    XY(2) = eps;
elseif XY(2) > 1-eps
   XY(2) = 1-eps;
end

XYtrunc = XY;