function [w,mse] = linear_regression_train(X,y)
    
    % normal equation or pseudo inverse
    % w = pinv(X'*X)*X'*y;
    X = featureNorm(X);
    num_iter = 1000;
    mse = zeros(num_iter,1);
    alpha = 2;
    [m,n] = size(X);
    X = [ones(m,1), X];
    w = zeros(n+1,1);
    for i=1:num_iter
        alpha = alpha/i;
        w = w + alpha * X'*((X * w)-y);
        mse(i) = (1/m) * sum((y - X*w).^2);
    end
end