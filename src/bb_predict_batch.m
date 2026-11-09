function y=bb_predict_batch(model,flight,override)
% Input-history inference; intentionally independent of measured outputs.
if nargin<3,override=struct();end
if model.outputLags~=0,error('Batch inference is for input-history models only');end
X=bb_lag_inputs(flight,model.inputNames,model.inputLags,override);
y=bb_forward(model,X);
end
