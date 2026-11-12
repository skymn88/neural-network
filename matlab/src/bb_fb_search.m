function [model,report,best]=bb_fb_search(tr,va,vDDV,cfg,betas,ridges)
if numel(cfg.oeWindowSeconds)>1
 best=Inf;report=table();model=[];
 for seconds=cfg.oeWindowSeconds(:)'
  c=cfg;c.oeWindowSeconds=seconds;
  [m,r,score]=bb_fb_search(tr,va,vDDV,c,betas,ridges);
  r.WindowSeconds=repmat(seconds,height(r),1);report=[report;r]; %#ok<AGROW>
  if isempty(model)||score<best,best=score;model=m;end
 end
 return;
end
% Fixed bank; only beta/ridge/lambda selected on validation, no test input.
best=Inf;model=[];rows={};c=cfg;c.oeFilterBank=true;c.oeZeroRateAtNull=false;
scales=1;if cfg.fbTanhSearch,scales=cfg.nlTanhScales;end
for scale=scales(:)'
c.oeTanhScale=scale;
for weight=unique([0 cfg.oeTransitionWeight(:)'])
 c.oeTransitionWeight=weight;
 for beta=betas
  for ridge=ridges
   fitTimer=tic;m=bb_oe_train(tr,c,beta,cfg.fbRawLags,ridge);seconds=toc(fitTimer);m=bb_ram_bounds(m,tr,cfg);
   [score,metric]=bb_ram_score(m,va,vDDV,cfg);rows(end+1,:)={beta,ridge,weight,numel(m.weights),seconds,score,scale,metric.FullFreeRunRMSE,metric.ResetWindowRMSE,metric.HoldDroop,metric.Eligible}; %#ok<AGROW>
   if isempty(model)||(isfinite(score)&&score<best),best=score;model=m;model.name='NL_FB';end
  end
 end
end
end
if isempty(model),error('No finite NL_FB validation model');end
report=cell2table(rows,'VariableNames',{'Beta','Ridge','TransitionWeight','Coefficients','TrainSeconds','ValidationRMSE','TanhScale','FullFreeRunRMSE','ResetWindowRMSE','HoldDroop','Eligible'});
report.WindowSeconds=repmat(cfg.oeWindowSeconds,height(report),1);
report.SelectionMode=repmat({cfg.selectionMode},height(report),1);
end
