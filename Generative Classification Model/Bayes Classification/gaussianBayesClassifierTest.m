function [mu,sd,trainError,testError] = gaussianBayesClassifierTest(train,test)
    n = size(train,2);
    trainX = train(:,1:n-1);
    trainY = train(:,n);
    
    n = size(test,2);
    testX = test(:,1:n-1);
    testY = test(:,n);
    
    params = gaussianBayesClassifier(train);
    
    mu = params{1};
    sd = params{2};
    
    class = zeros(size(testY,1),1);
    
    likelihood0 = prod(mvnpdf(testX,mu(1,:),sd(1,:)),2);
    prior0 = sum(testY == 0)/length(testY);
    
    likelihood1 = prod(mvnpdf(testX,mu(2,:),sd(2,:)),2);
    prior1 = sum(testY == 1)/length(testY);
    
    class(likelihood0*prior0 > likelihood1*prior1) = 0;
    class(likelihood0*prior0 <= likelihood1*prior1) = 1;
    
    testError = sum(testY ~= class)/size(testY,1);
    
    class = zeros(size(trainY,1),1);
    
    likelihood0 = prod(mvnpdf(trainX,mu(1,:),sd(1,:)),2);
    prior0 = sum(trainY == 0)/length(trainY);
    
    likelihood1 = prod(mvnpdf(trainX,mu(2,:),sd(2,:)),2);
    prior1 = sum(trainY == 1)/length(trainY);
    
    class(likelihood0*prior0 > likelihood1*prior1) = 0;
    class(likelihood0*prior0 <= likelihood1*prior1) = 1;
    
    trainError = sum(trainY ~= class)/size(trainY,1);
end