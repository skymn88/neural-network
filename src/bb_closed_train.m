function [closed,selection]=bb_closed_train(tr,va,plant,cfg,lags,betas,ridges)
% Isolated closed-mode selection: validation free run, never test.
lo=Inf;hi=-Inf;for f=1:numel(tr),lo=min(lo,min(tr(f).S.RAM));hi=max(hi,max(tr(f).S.RAM));end
plant.trainBounds=[lo hi];closed.plant=plant;rows={};best=Inf;closed.controller=[];
for tau=cfg.ctrlTaus
 for ridge=cfg.ctrlRidge
  c=bb_ctrl_train(tr,cfg,tau,ridge);sse=0;n=0;diverged=false;
  for f=1:numel(va)
   [p,info]=bb_closed_loop(c,plant,va(f),va(f).S.RAM(1));ix=cfg.evalStartIndex:numel(p);
   sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);diverged=diverged||info.diverged;
  end
  score=sqrt(sse/n);if diverged||~isfinite(score),score=Inf;end
  rows(end+1,:)={'Hybrid',tau,NaN,NaN,ridge,score,diverged}; %#ok<AGROW>
  if isempty(closed.controller),closed.controller=c;end
  if score<best,best=score;closed.controller=c;end
 end
end
closed.hybridValidationRMSE=best;
if ~isfinite(best),warning('bb:ClosedDivergence','No acceptable hybrid validation rollout; reporting first candidate as failed.');end
sc=cfg;sc.usePressure=false;sc.oeZeroRateAtNull=false;sc.oeInputName='RAMSetpoint';bestSP=Inf;
for beta=betas(betas<0)
 for lag=lags
  for ridge=ridges
   m=bb_oe_train(tr,sc,beta,lag,ridge);m=bb_ram_bounds(m,tr,cfg);
   m.divergenceThreshold=max(eps,2*max(abs(plant.trainBounds)));sse=0;n=0;diverged=false;
   for f=1:numel(va)
    [p,~,info]=bb_rollout_rate(m,va(f),va(f).S.RAM(1));diverged=diverged||info.diverged;ix=cfg.evalStartIndex:numel(p);
    diverged=diverged||any(~isfinite(p))||any(abs(p)>max(eps,2*max(abs(plant.trainBounds))));
    sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
   end
   score=sqrt(sse/n);if diverged||~isfinite(score),score=Inf;end
   rows(end+1,:)={'SP_OE',NaN,beta,lag,ridge,score,diverged}; %#ok<AGROW>
   if ~isfield(closed,'setpoint'),closed.setpoint=m;end
   if score<bestSP,bestSP=score;closed.setpoint=m;end
  end
 end
end
if ~isfield(closed,'setpoint'),error('SP_OE needs at least one negative beta candidate');end
closed.accepted=isfinite(min(best,bestSP));
if ~isfinite(bestSP),warning('bb:SPFailure','No acceptable SP_OE validation rollout.');end
closed.spValidationRMSE=bestSP;closed.selected='SP_OE';if best<bestSP,closed.selected='Hybrid';end
selection=cell2table(rows,'VariableNames',{'Model','Tau','Beta','Lags','Ridge','ValidationRMSE','Diverged'});
end
