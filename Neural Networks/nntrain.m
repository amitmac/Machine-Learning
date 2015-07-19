function [w_1 w_2 w_3] = nntrain(xtrain,ytrain,hidden_layer_size_1,hidden_layer_size_2,alpha)
    
    % Neural Network with 2 hidden layers %
    % xtrain = n x m , ytrain = n x k %
    [n,m] = size(xtrain);
    
    w_1 = InitializeWeights(hidden_layer_size_1, m + 1); 
    w_2 = InitializeWeights(hidden_layer_size_2, hidden_layer_size_1 + 1); 
    w_3 = InitializeWeights(k, hidden_layer_size_2 + 1); 
    
    for data_index = 1:n
        
        % Forward Propagation %
        act_func_1 = xtrain(data_index,:)';
        act_func_2 = sigmoid(w_1 * [1; act_func_1]);
        act_func_3 = sigmoid(w_2 * [1; act_func_2]);
        act_func_4 = sigmoid(w_3 * [1; act_func_3]);
        
        % Back Propagation %
        delta_4 = act_func_4 - ytrain(data_index,:)';
        
        delta_3 = (delta_4 .* (act_func_4 .* (1 - act_func_4)));
        w_3(:,2:hidden_layer_size_2 + 1) = w_3(:,2:hidden_layer_size_2 + 1) - alpha * delta_3 * act_func_3';
        
        delta_2 = (delta_3' * w_2(:,2:hidden_layer_size_1 + 1))' .* (act_func_3 .* (1 - act_func_2));
        w_2(:,2:hidden_layer_size_1 + 1) = w_2(:,2:hidden_layer_size_1 + 1) - alpha * delta_2 * act_func_2';
        
        delta_1 = (delta_2' * w_1(:,2:m + 1))' .* (act_func_2 .* (1 - act_func_2));
        w_1(:,2:m + 1) = w_1(:,2:m + 1) - alpha * delta_1 * act_func_1';
        
    end
    
end