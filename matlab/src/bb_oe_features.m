function X=bb_oe_features(model,flight,idx)
% Static scalar DDV map is applied BEFORE delays; no target/position access.
idx=idx(:);names=model.inputNames;
if isfield(model,'filterBank')
 U=zeros(max(idx),numel(names));
 for j=1:numel(names),U(:,j)=flight.S.(names{j})(1:max(idx));end
 X=bb_filter_bank(U,model.filterBank);X=X(idx,:);return;
end
if ~isfield(model,'featureSpec')
 X=zeros(numel(idx),(model.inputLags+1)*numel(names));
 for lag=0:model.inputLags
  for j=1:numel(names)
   v=flight.S.(names{j});X(:,lag*numel(names)+j)=v(max(1,idx-lag));
  end
 end
 return;
end
spec=model.featureSpec;X=zeros(numel(idx),model.inputLags+1);u=flight.S.DDV;
for lag=0:model.inputLags
 z=u(max(1,idx-lag));
 switch spec.basis
  case {'deadband','hold'},z=sign(z).*max(abs(z)-spec.parameter,0);
  case 'tanh',z=tanh(z/spec.parameter);
  case {'linear','nonlinear'}
  otherwise,error('Unknown static input map');
 end
 X(:,lag+1)=z;
end
if spec.force||spec.pressure
 for lag=0:8
  ii=max(1,idx-lag);z=u(ii);
  if spec.force
   if ~isfield(flight.S,'ActuatorForce'),error('ActuatorForce required by _F model');end
   force=flight.S.ActuatorForce(ii);
   if ~model.noIntercept,X=[X force];end %#ok<AGROW>
   X=[X z.*force]; %#ok<AGROW>
  end
  if spec.pressure
   if ~isfield(flight.S,'SupplyPressure')||~isfield(flight.S,'ReturnPressure'),error('Pressure channels required by _P model');end
   dp=flight.S.SupplyPressure(ii)-flight.S.ReturnPressure(ii);
   if any(dp<0),error('Supply pressure below return pressure: check units/alignment');end
   X=[X z.*sqrt(dp)]; %#ok<AGROW>
  end
 end
end
end
