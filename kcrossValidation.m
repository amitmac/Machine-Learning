function [accuracy,errorbar,parameters] = kcrossValidation(learningModel,data,k)
    [m,n] = size(data);
    indices = crossvalind('Kfold',m,k);
    accuracies = zeros(k,1);
    for i=1:k
        train = data(indices ~= i,:);
        test = data(indices == i,:);
        [calculatedClasses,params] = learningModel(train,test);
        trueClasses = test(:,n);
        accuracies(i) = length(find(trueClasses == calculatedClasses))/m;
    
        if(i==1)
            parameters = params;
        else
            for j = 1:length(params)
                parameters{j} = parameters{j} + params{j};
            end
        end
    end
    for i=1:k
        parameters{i} = parameters{i}/k;
    end
    accuracy = mean(accuracies);
    errorbar = std(accuracies);
end