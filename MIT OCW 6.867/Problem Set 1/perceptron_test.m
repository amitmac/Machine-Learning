function test_error = perceptron_test(X,y,theta)
    n = size(X,1);
    X = [ones(n,1),X];
	hypothesis = X*theta;
	error=0;
	for i=1:size(X,1)
		if(transpose(y)*hypothesis < 0)
			error = error + 1;
		end
	end
	test_error = error*100/size(X,1); 
end