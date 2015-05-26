function theta = logistic_regression_train(X,y,alpha,num_iter)
	
	X = featureNorm(X);
    [m,n] = size(X);
    X = [ones(m,1), X];
    theta = zeros(n+1,1);
	
    for i=1:num_iter
        alpha = alpha/i;
        h = sigmoid(X * w);
        theta = theta + alpha * X' * (h - y);
    end

end