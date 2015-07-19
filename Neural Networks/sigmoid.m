function h = sigmoid(z)
    h = 1.0 ./(1.0+ exp(-z));
end