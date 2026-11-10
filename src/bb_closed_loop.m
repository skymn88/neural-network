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
ddvBank=[];ramBank=[];previous=x0;diverged=false;threshold=2*max(abs(plant.trainBounds));threshold=max(threshold,eps);
for k=1:N
 d=0;i=0;if k>1,d=ddv(k-1);i=current(k-1);end
 [z,state]=bb_ctrl_features(flight.S.RAMSetpoint(k)-previous,d,i,state,controller);
 current(k)=bb_forward(controller,z);
 if ~isfinite(current(k)),diverged=true;break;end
 buffer.S.CoilCurrent=[buffer.S.CoilCurrent(2:end);current(k)];
 [ddv(k),ddvBank]=localStep(plant.ddv,buffer,L,ddvBank);
 buffer.S.DDV=[buffer.S.DDV(2:end);ddv(k)];
 for j=1:numel(aux)
  if isfield(buffer.S,aux{j}),buffer.S.(aux{j})=[buffer.S.(aux{j})(2:end);flight.S.(aux{j})(k)];end
 end
 [rate,ramBank]=localStep(plant.rate,buffer,L,ramBank);
 if isfield(plant.rate,'rateGain'),rate=rate*plant.rate.rateGain;end
 if isfield(plant.rate,'rateBias'),rate=rate+plant.rate.rateBias;end
 if isfield(plant.rate,'wh'),rate=bb_wh_map(plant.rate.wh,rate);end
 next=previous;if k>1,next=bb_ram_step(plant.rate,previous,rate);end
 if ~isfinite(next)||abs(next)>threshold,diverged=true;end
 if ~isfinite(next)||~isfinite(ddv(k)),break;end
 if isfield(plant.rate,'ramBounds'),next=min(max(next,plant.rate.ramBounds(1)),plant.rate.ramBounds(2));end
 x(k)=next;previous=next;
end
info=struct('diverged',diverged,'current',current,'ddv',ddv,'threshold',threshold);
end
function [y,state]=localStep(model,buffer,last,state)
if isfield(model,'filterBank')
 U=zeros(1,numel(model.inputNames));
 for j=1:numel(model.inputNames),U(j)=buffer.S.(model.inputNames{j})(last);end
 [X,state]=bb_filter_bank(U,model.filterBank,state);
else,X=bb_oe_features(model,buffer,last);end
y=bb_forward(model,X);
end
