function w = InitializeWeights(L_in, L_out)
    % L_in is number of incoming connections
    % L_out is number of outgoing connections
    e_init = sqrt(6)/sqrt(L_in + L_out);
    w = rand(L_out, 1 + L_in) * 2 * e_init - e_init;
end