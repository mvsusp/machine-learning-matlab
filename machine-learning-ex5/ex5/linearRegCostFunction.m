function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples


h = X * theta;
cost = sum((h - y) .^ 2);

l_theta = [zeros(1,size(theta,2)) ; theta(2:end, :)];

regularization = lambda * sum(sum(l_theta .^ 2))/(2*m);

J = cost/(2*m) + regularization;


grad = X'*(h-y)/m;
grad_reg = l_theta * lambda / m;

grad = grad + grad_reg;

grad = grad(:);

end
