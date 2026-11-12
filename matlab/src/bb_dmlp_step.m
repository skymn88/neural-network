function next=bb_dmlp_step(model,x,u)
% Batched single-state neural increment, portable base MATLAB inference.
q=model.dynamicMLP;z=[u (x-q.xCenter)/q.xScale];
h=tanh(bsxfun(@plus,z*q.W',q.b'));
rate=(h*q.v+q.bias)*q.rateScale;
next=x+rate/model.Fs;
end
