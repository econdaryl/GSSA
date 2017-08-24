function XYtrunc = test_trunc(XY)
% XY is an nx+ny by 1 vector

% XY(1) is capital

eps = .000001;

% capital cannot be negative
if XY(1) < eps
    XY(1) = eps;
end

XYtrunc = XY;