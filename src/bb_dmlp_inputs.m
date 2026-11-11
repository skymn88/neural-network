function U=bb_dmlp_inputs(X)
% Causal external features from DDV lag 0..8; no measured RAM access.
U=[X(:,1) mean(X,2) X(:,1)-X(:,2)];
end
