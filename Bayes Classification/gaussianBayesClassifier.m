function params = gaussianBayesClassifier(train)
    n = size(train,2);
    X = train(:,1:n-1);
    Y = train(:,n);
    
    % Segment the data according to target/class and process
    uniqueY = unique(Y);
    mu = zeros(length(uniqueY),size(X,2));
    sd = zeros(length(uniqueY),size(X,2));
    
    for i = 1:length(uniqueY)
        d = X(uniqueY(i) == Y,:);
        mu(i,:) = mean(d);
        sd(i,:) = std(d);
    end
    params = {mu,sd};
end