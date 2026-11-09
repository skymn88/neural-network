function [x,info]=bb_closed_loop(controller,plant,flight,x0)
% Frozen DDV/RAM plant. Only setpoint + optional external force/pressure + x0.
if ~isfield(flight.S,'RAMSetpoint'),error('RAMSetpoint required for closed loop');end
if ~isscalar(x0)||~isfinite(x0),error('Initial RAM must be finite scalar');end
if controller.Fs~=plant.rate.Fs||plant.ddv.Fs~=plant.rate.Fs,error('Controller/plant sample rates differ');end
N=numel(flight.t);x=nan(N,1);current=nan(N,1);ddv=nan(N,1);state=[];
L=max([plant.ddv.inputLags plant.rate.inputLags 8])+1;
buffer.t=(0:L-1)'/plant.rate.Fs;buffer.S.CoilCurrent=zeros(L,1);buffer.S.DDV=zeros(L,1);
aux={'ActuatorForce','SupplyPressure','ReturnPressure'};
for j=1:numel(aux)
 if isfield(flight.S,aux{j}),buffer.S.(aux{j})=repmat(flight.S.(aux{j})(1),L,1);end
end
previous=x0;diverged=false;threshold=2*max(abs(plant.trainBounds));threshold=max(threshold,eps);
for k=1:N
 d=0;i=0;if k>1,d=ddv(k-1);i=current(k-1);end
 [z,state]=bb_ctrl_features(flight.S.RAMSetpoint(k)-previous,d,i,state,controller);
 current(k)=bb_forward(controller,z);
 if ~isfinite(current(k)),diverged=true;break;end
 buffer.S.CoilCurrent=[buffer.S.CoilCurrent(2:end);current(k)];
 ddv(k)=localStep(plant.ddv,buffer,L);
 buffer.S.DDV=[buffer.S.DDV(2:end);ddv(k)];
 for j=1:numel(aux)
  if isfield(buffer.S,aux{j}),buffer.S.(aux{j})=[buffer.S.(aux{j})(2:end);flight.S.(aux{j})(k)];end
 end
 rate=localStep(plant.rate,buffer,L);
 if isfield(plant.rate,'rateGain'),rate=rate*plant.rate.rateGain;end
 if isfield(plant.rate,'rateBias'),rate=rate+plant.rate.rateBias;end
 next=previous;if k>1,next=(1+plant.rate.beta/plant.rate.Fs)*previous+rate/plant.rate.Fs;end
 if ~isfinite(next)||abs(next)>threshold,diverged=true;end
 if ~isfinite(next)||~isfinite(ddv(k)),break;end
 if isfield(plant.rate,'ramBounds'),next=min(max(next,plant.rate.ramBounds(1)),plant.rate.ramBounds(2));end
 x(k)=next;previous=next;
end
info=struct('diverged',diverged,'current',current,'ddv',ddv,'threshold',threshold);
end
function y=localStep(model,buffer,last)
X=bb_oe_features(model,buffer,last);y=bb_forward(model,X);
end
