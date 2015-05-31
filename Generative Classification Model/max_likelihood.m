function [mu,covar,prior] = max_likelihood(X,Y)
    uniqueY = unique(Y);
    n = length(uniqueY);
    mu = zeros(n,size(X,2));
    prior = zeros(n,1);
    covar = zeros(size(X,2),size(X,2));
    for i=1:n
        mu(i,:) = mean(X(uniqueY(i)==Y,:));
        prior(i) = sum(uniqueY(i)==Y) / length(X);
        covar = covar + (size(Y(uniqueY(i) == Y),1)*cov(X(uniqueY(i)==Y,:)));
    end
    covar = covar ./ size(Y,1);
end