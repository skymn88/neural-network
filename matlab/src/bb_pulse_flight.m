function fl=bb_pulse_flight(tr,cfg)
% Synthetic stimulus scaled from TRAIN only; never a physical reference target.
p=cfg.pulseProfile;
if size(p,2)~=2||any(~isfinite(p(:)))||any(p(:,2)<=0),error('pulseProfile must be [std-multiplier seconds] rows');end
v=[];x=[];for f=1:numel(tr),v=[v;tr(f).S.CoilCurrent];x=[x;tr(f).S.RAM];end %#ok<AGROW>
u=[];segments=[];
for j=1:size(p,1)
 n=max(1,round(p(j,2)*cfg.Fs));u=[u;repmat(p(j,1)*std(v),n,1)];segments=[segments;repmat(j,n,1)]; %#ok<AGROW>
end
fl.name='TRAIN-scaled synthetic pulse';fl.t=(0:numel(u)-1)'/cfg.Fs;
fl.S.CoilCurrent=u;fl.S.RAM=repmat(mean(x),numel(u),1);fl.pulseSegment=segments;fl.pulseProfile=p;
aux={'ActuatorForce','SupplyPressure','ReturnPressure'};
for j=1:numel(aux)
 if isfield(tr(1).S,aux{j})
  v=[];for f=1:numel(tr),v=[v;tr(f).S.(aux{j})];end %#ok<AGROW>
  fl.S.(aux{j})=repmat(mean(v),numel(u),1);
 end
end
end
