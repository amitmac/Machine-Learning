function [a, b] = svmtrain(X,y,fn,C)
    
    n = size(X,1);
    
    % convert y from {0,1} to {-1,1}
    y = 2*y - 1; 
    
    %   SVM optimization condition
    %   
    %   maximise 
    %       L(a) = sum_{i=1}^{n}(a_i) - (1/2)sum_{i=1}^{n}sum_{j=1}^{n}(a_i)(a_j)(y_i)(y_j)(X_i)(X_j)
    %   subject to 
    %       a_i >= 0 and sum_{i=1}^{n}(a_i)(y_i)
    %
    %   Let H_{ij} = (y_i)(y_j)(X_i)(X_j)
    %       A = [a_1 a_2 a_3 ... a_n]
    %
    %   L(a) = sum_{i=1}^{n}(a_i) - (1/2)A'*H*A
    %
    %   We use quadprog function of matlab which solves the quadratic
    %   problem of form
    %
    %   maximise L(a) = (1/2)a'*H*a + f'a 
    %   subject to Aa <= C and Ba = D
    
    H = zeros(n);
    for i = 1:n
        for j = 1:n
            H(i,j) = y(i) * y(j) * fn(X(i,:)' * X(j,:)');
        end
    end
    
    f = -ones(n,1);
    A = y';
    
    a = quadprog(H,f,[],[],A,0,zeros(n,1),C*ones(n,1));
    
    %   Decision Rule:
    %       sum(a_i)(y_i)(K(x_i,x_j)) + b >= 0 ==> Class 1
    %       sum(a_i)(y_i)(K(x_i,x_j)) + b < 0  ==> Class 2
    %   
    %   To calculate b, we consider set of indices for which
    %   0 < a_n < C.
    %   b = (1/l)sum(y_n - sum((a_m)(y_m)(K(x_n,x_m))) )
    
    b = 0;
    l = length(find(a > 0 & a < C));
    for i = 1:n
        if a(i) > 0 && a(i) < C
            b = b + y(i);
            for j = 1:n
                if a(j) > 0
                    b = b - a(j)*y(j)*func(X(j,:)',X(i,:)');
                end
            end
        end
    end
    b = b / l;
end