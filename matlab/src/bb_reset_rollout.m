function [x,mask]=bb_reset_rollout(m,fl,ddv,seconds,start)
% Measured RAM resets are diagnostic only. External input/filter history persists.
if ~isscalar(seconds)||~isfinite(seconds)||seconds<=0,error('Reset horizon must be finite and positive');end
N=numel(fl.t);drive=bb_predict_batch(m,fl,struct('DDV',ddv));
if isfield(m,'rateGain'),drive=m.rateGain*drive;end
if isfield(m,'rateBias'),drive=drive+m.rateBias;end
if isfield(m,'wh'),drive=bb_wh_map(m.wh,drive);end
W=max(2,round(seconds*m.Fs));x=nan(N,1);mask=false(N,1);
for b=start:W:N
 last=min(N,b+W-1);x(b)=fl.S.RAM(b);
 linear=~isfield(m,'dynamicMLP')&&~isfield(m,'stateSpace')&&~isfield(m,'stateLeak')&&~isfield(m,'constantRAM')&&~isfield(m,'ramBounds');
 if linear
  ix=b:last;u=drive(ix);u(1)=0;a=1+m.beta/m.Fs;
  x(ix)=fl.S.RAM(b)*a.^((0:numel(ix)-1)')+filter(1/m.Fs,[1 -a],u);
 else
 for k=b+1:last
  x(k)=bb_ram_step(m,x(k-1),drive(k,:));
  if isfield(m,'ramBounds'),x(k)=min(max(x(k),m.ramBounds(1)),m.ramBounds(2));end
 end
 end
 mask(b+1:last)=true;
end
end
