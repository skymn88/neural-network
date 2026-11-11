function y=bb_predict_batch(model,flight,override)
% Input-history inference; intentionally independent of measured outputs.
if nargin<3,override=struct();end
if isfield(model,'ddvCenter')
 if isfield(override,'DDV'),override.DDV=override.DDV-model.ddvCenter;
 else,override.DDV=flight.S.DDV-model.ddvCenter;end
end
if model.outputLags~=0,error('Batch inference is for input-history models only');end
if isfield(model,'featureSpec')||isfield(model,'filterBank')
 names=fieldnames(override);
 for j=1:numel(names),flight.S.(names{j})=override.(names{j});end
 X=bb_oe_features(model,flight,(1:numel(flight.t))');
else
 X=bb_lag_inputs(flight,model.inputNames,model.inputLags,override);
end
y=bb_forward(model,X);
end
