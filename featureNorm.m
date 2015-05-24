function normX = featureNorm(X)
    m = length(X);
    means = mean(X);
    sigma = std(X);
    normX = (X - repmat(means,m,1)) ./ repmat(sigma,m,1);
end