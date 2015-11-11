function [J_history] = converge(X, y)


num_iters = 1500;
alpha = 0.01;
theta = zeros(2, 1);
m = length(y); 
J_history = zeros(num_iters, 1);
theta_val = zeros(num_iters,1);

for iter = 1:num_iters
    
    h = X * theta;
    e = h -y;
    theta_change = (alpha / m)* X'*e;
    theta = theta - theta_change; 
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
