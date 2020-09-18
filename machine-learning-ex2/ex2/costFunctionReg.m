function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hypo=sigmoid(X*theta);
y1=-y.*log(hypo);
y0=(ones(size(y))-y).*log(ones(size(hypo))-hypo);
regJ=sum(theta(2:end).^2) * lambda / (2*m);
J=(sum(y1-y0+regJ))/m;

grad1 = (X' * (hypo - y)) ./ m;
regG = theta(2:end) .* lambda ./ m;
regG = [0; regG];
grad=grad1+regG;



% =============================================================

end
