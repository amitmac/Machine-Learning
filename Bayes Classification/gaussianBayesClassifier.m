function class = naiveBayesClassifier(train,test)
    [m,n] = size(train);
    X = train(:,1:n-1);
    Y = train(:,n);
    
    % Segment the data according to target/class and process
    uniqueY = unique(Y);
    class = zeros(size(test));
    for i = 1:length(uniqueY)
        d = X(uniqueY(i) == train(:,n));
        mu = mean(d);
        vars = var(d);
        likelihood = zeros(size(test));
        for j = 1:length(test)
            % Suppose likelihood is given by normal distribution
            likelihood(i,:) = normpdf(test(i,:),mu,vars);
        end
        likelihood = prod(likelihood,2);
        prior = length(d)/m;
        
        class = (likelihood*prior) > class;
    end
end