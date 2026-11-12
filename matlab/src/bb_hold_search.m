function [model,report,best]=bb_hold_search(tr,va,vDDV,cfg)
if numel(cfg.oeWindowSeconds)>1
 best=Inf;report=table();model=[];
 for seconds=cfg.oeWindowSeconds(:)'
  c=cfg;c.oeWindowSeconds=seconds;
  [m,r,score]=bb_hold_search(tr,va,vDDV,c);
  r.WindowSeconds=repmat(seconds,height(r),1);report=[report;r]; %#ok<AGROW>
  if isempty(model)||score<best,best=score;model=m;end
 end
 return;
end
% Learned asymmetric increments with finite memory and an exact integrator.
% No prescribed motion amplitude, direction, or return position.
v=[];for f=1:numel(tr),v=[v;tr(f).S.DDV];end %#ok<AGROW>
scale=max(std(v),1e-12);rows={};best=Inf;model=[];
c=cfg;c.oeCustom=true;c.oeZeroRateAtNull=true;c.oeTransitionWeight=0;
c.oeFilterBank=false;c.oeNonlinear=true;
betas=0;if cfg.oeWindowSearch,betas=cfg.oeWindowBetas;end
for beta=betas(:)'
for lag=cfg.holdLags
 for fraction=cfg.holdDeadbandFractions
  if fraction<0||lag<0||lag~=floor(lag),error('Invalid hold candidate settings');end
  c.featureSpec=struct('basis','hold','parameter',fraction*scale,'force',cfg.useForce,'pressure',cfg.usePressure);
  for ridge=cfg.holdRidge
   timer=tic;m=bb_oe_train(tr,c,beta,lag,ridge);m=bb_ram_bounds(m,tr,cfg);
   [score,metric]=bb_ram_score(m,va,vDDV,cfg);rows(end+1,:)={beta,lag,ridge,fraction*scale,score,toc(timer),metric.FullFreeRunRMSE,metric.ResetWindowRMSE,metric.HoldDroop,metric.Eligible}; %#ok<AGROW>
   if isempty(model)||(isfinite(score)&&score<best),best=score;model=m;model.name='NL_HOLD';end
  end
 end
end
end
if isempty(model),error('No finite NL_HOLD validation model');end
report=cell2table(rows,'VariableNames',{'Beta','Lags','Ridge','Deadband','ValidationRMSE','Seconds','FullFreeRunRMSE','ResetWindowRMSE','HoldDroop','Eligible'});
report.WindowSeconds=repmat(cfg.oeWindowSeconds,height(report),1);
report.SelectionMode=repmat({cfg.selectionMode},height(report),1);
end
