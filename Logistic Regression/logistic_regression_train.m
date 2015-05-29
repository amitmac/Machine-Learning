function [theta,cost] = logistic_regression_train(X,y,num_iter)
	
	%X = featureNorm(X);
    [m,n] = size(X);
    X = [ones(m,1), X];
    theta = ones(n+1,1);
	cost = zeros(1,500);
    for i=1:num_iter
        alpha = 1/sqrt(i); %try changing different values like 1/sqrt(i),2/i...
        h = sigmoid(X * theta);
        theta = theta - alpha * X' * (h - y);
        cost(i) = sum(abs(h - y)) / m;
    end

end