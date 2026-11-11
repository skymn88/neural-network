function [model,report,best]=bb_hold_search(tr,va,vDDV,cfg)
% Learned asymmetric increments with finite memory and an exact integrator.
% No prescribed motion amplitude, direction, or return position.
v=[];for f=1:numel(tr),v=[v;tr(f).S.DDV];end %#ok<AGROW>
scale=max(std(v),1e-12);rows={};best=Inf;model=[];
c=cfg;c.oeCustom=true;c.oeZeroRateAtNull=true;c.oeTransitionWeight=0;
c.oeFilterBank=false;c.oeNonlinear=true;
for lag=cfg.holdLags
 for fraction=cfg.holdDeadbandFractions
  if fraction<0||lag<0||lag~=floor(lag),error('Invalid hold candidate settings');end
  c.featureSpec=struct('basis','hold','parameter',fraction*scale,'force',cfg.useForce,'pressure',cfg.usePressure);
  for ridge=cfg.holdRidge
   timer=tic;m=bb_oe_train(tr,c,0,lag,ridge);m=bb_ram_bounds(m,tr,cfg);
   sse=0;n=0;
   for f=1:numel(va)
    p=bb_rollout_rate(m,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=cfg.evalStartIndex:numel(p);
    sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
   end
   score=sqrt(sse/n);rows(end+1,:)={0,lag,ridge,fraction*scale,score,toc(timer)}; %#ok<AGROW>
   if isfinite(score)&&score<best,best=score;model=m;model.name='NL_HOLD';end
  end
 end
end
if isempty(model),error('No finite NL_HOLD validation model');end
report=cell2table(rows,'VariableNames',{'Beta','Lags','Ridge','Deadband','ValidationRMSE','Seconds'});
end
