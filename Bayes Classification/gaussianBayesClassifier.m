function class = gaussianBayesClassifier(train,test)
    [m,n] = size(train);
    X = train(:,1:n-1);
    Y = train(:,n);
    
    % Segment the data according to target/class and process
    uniqueY = unique(Y);
    test = test(:,1:n-1);
    class = zeros(size(test,1),1);
    for i = 1:length(uniqueY)
        d = X(uniqueY(i) == train(:,n));
        mu = mean(d);
        vars = var(d);
        likelihood = zeros(size(test));
        for j = 1:length(test)
            % Suppose likelihood is given by normal distribution
            likelihood(i,:) = normpdf(test(i,:),mu,vars);
        end
        plikelihood = prod(likelihood,2);
        prior = length(d)/m;
        
        class(plikelihood*prior > class) = uniqueY(i);
    end
end