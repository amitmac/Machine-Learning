function [trainError,testError] = logistic_regression_test(trainX,trainY,testX,testY,theta)
	
    trainX = [ones(size(trainX,1),1), trainX];
    testX = [ones(size(testX,1),1),testX];
    
    calctrainY = sigmoid(trainX * theta) >= 0.5;
	trainError = sum(abs(calctrainY - trainY)) / length(trainX);
    
    calctestY = sigmoid(testX * theta) >= 0.5;
	testError = sum(abs(calctestY - testY)) / length(testX);
    
end