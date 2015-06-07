function [theta,k,gamma_geom] = perceptron_train(X,y)
    [n,d] = size(X);
    theta = zeros(d+1,1);
    k=0;
    X = [ones(n,1),X];
    theta = theta + y(1)*transpose(X(1,:));
    num_correct = 0;
    index = 2;
    while(num_correct < n)
	for j=index:n
	    if ((X(j,:)*theta)*y(j) < 0)
	        k = k + 1;
                num_correct = 0;
		theta = theta + (y(j)*transpose(X(j,:)));
            else
                num_correct = num_correct + 1;
	    end
        end
        index = 1;
    end
    
    % geometric margin
    gamma_geom = min(abs(X*theta / norm(theta)));
end
