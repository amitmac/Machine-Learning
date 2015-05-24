function mse = linear_regression_test(Xtest,ytest,w)

    [m,n] = size(Xtest);
    Xtest = [ones(m,1), Xtest];
    ycalc = Xtest * w;
    mse = sum((ytest - ycalc).^2)/m;

end