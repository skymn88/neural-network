function y=bb_predict_batch(model,flight,override)
% Input-history inference; intentionally independent of measured outputs.
if nargin<3,override=struct();end
if model.outputLags~=0,error('Batch inference is for input-history models only');end
if isfield(model,'featureSpec')
 names=fieldnames(override);
 for j=1:numel(names),flight.S.(names{j})=override.(names{j});end
 X=bb_oe_features(model,flight,(1:numel(flight.t))');
else
 X=bb_lag_inputs(flight,model.inputNames,model.inputLags,override);
end
y=bb_forward(model,X);
end
