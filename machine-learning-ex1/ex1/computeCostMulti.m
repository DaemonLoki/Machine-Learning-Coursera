function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
h0 = (X * theta);
diffSquared = (h0 - y).^2;
J = sum(diffSquared) / (2*m);

% =========================================================================

end
