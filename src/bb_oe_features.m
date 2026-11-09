function X=bb_oe_features(model,flight,idx)
% Bounded-memory causal raw features; repeated initial input before time zero.
idx=idx(:);names=model.inputNames;X=zeros(numel(idx),(model.inputLags+1)*numel(names));
for lag=0:model.inputLags
 for j=1:numel(names)
  v=flight.S.(names{j});X(:,lag*numel(names)+j)=v(max(1,idx-lag));
 end
end
end
