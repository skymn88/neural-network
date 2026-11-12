function [bestModel,report,best]=bb_oe_search(tr,va,vDDV,cfg,label,lags,betas,ridges)
% Fixed split; validation ONLY selects static map and OE hyperparameters.
best=Inf;bestModel=[];rows={};specs={struct('basis','linear','parameter',0,'force',false,'pressure',false)};
if strncmp(label,'NL_OE',5),specs{1}.basis='nonlinear';lags=lags(lags<=128);end
if strncmp(label,'HAM_OE',6)
 specs={};lags=lags(lags<=128);
 for d=cfg.hamDeadbands
  if d<0,error('Negative deadband');end
  specs{end+1}=struct('basis','deadband','parameter',d,'force',false,'pressure',false); %#ok<AGROW>
 end
 for s=cfg.hamScales
  if s<=0,error('Nonpositive tanh scale');end
  specs{end+1}=struct('basis','tanh','parameter',s,'force',false,'pressure',false); %#ok<AGROW>
 end
end
for j=1:numel(specs)
 c=cfg;c.oeTransitionWeight=0;c.oeCustom=true;c.oeZeroRateAtNull=false;c.featureSpec=specs{j};
 c.featureSpec.force=~isempty(strfind(label,'_F')); %#ok<STREMP>
 c.featureSpec.pressure=~isempty(strfind(label,'_P')); %#ok<STREMP>
 for beta=betas
  for lag=lags
   for ridge=ridges
    m=bb_oe_train(tr,c,beta,lag,ridge);m=bb_ram_bounds(m,tr,cfg);sse=0;n=0;
    for f=1:numel(va)
     p=bb_rollout_rate(m,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=cfg.evalStartIndex:numel(p);
     sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
    end
    score=sqrt(sse/n);rows(end+1,:)={label,specs{j}.basis,specs{j}.parameter,beta,lag,ridge,score}; %#ok<AGROW>
    if isfinite(score)&&score<best,best=score;bestModel=m;bestModel.name=label;end
   end
  end
 end
end
if isempty(bestModel),error('No finite validation candidate: %s',label);end
report=cell2table(rows,'VariableNames',{'Model','Basis','Parameter','Beta','Lags','Ridge','ValidationRMSE'});
end
