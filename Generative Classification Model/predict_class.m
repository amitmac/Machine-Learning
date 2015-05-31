function [trainError,testError] = predict_class(trainX,trainY,testX,testY,mu,covar,prior)
    
    class = zeros(size(testY,1),1);

    loglikelihood0 = log(mvnpdf(testX,mu(1,:),covar));
    logprior0 = log(prior(1));

    loglikelihood1 = log(mvnpdf(testX,mu(2,:),covar));
    logprior1 = log(prior(2));

    class(loglikelihood0 + logprior0 > loglikelihood1 + logprior1) = 0;
    class(loglikelihood0 + logprior0 <= loglikelihood1 + logprior1) = 1;
    
    testError = sum(testY ~= class)/size(testY,1);
    
    class = zeros(size(trainY,1),1);
    
    loglikelihood0 = log(mvnpdf(trainX,mu(1,:),covar));
    logprior0 = log(prior(1));

    loglikelihood1 = log(mvnpdf(trainX,mu(2,:),covar));
    logprior1 = log(prior(2));

    class(loglikelihood0 + logprior0 > loglikelihood1 + logprior1) = 0;
    class(loglikelihood0 + logprior0 <= loglikelihood1 + logprior1) = 1;
    
    trainError = sum(trainY ~= class)/size(trainY,1);
end