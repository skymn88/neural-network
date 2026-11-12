function [bestModel,report,best,weighted]=bb_nl_search(tr,va,vDDV,cfg,lags,betas,ridges)
if numel(cfg.oeWindowSeconds)>1
 best=Inf;report=table();bestModel=[];
 weighted=struct();
 for seconds=cfg.oeWindowSeconds(:)'
  c=cfg;c.oeWindowSeconds=seconds;
  [m,r,score,w]=bb_nl_search(tr,va,vDDV,c,lags,betas,ridges);
  fields=fieldnames(w);
  for wi=1:numel(fields),key=matlab.lang.makeValidName(sprintf('%s_window_%g',fields{wi},seconds));weighted.(key)=w.(fields{wi});end
  r.WindowSeconds=repmat(seconds,height(r),1);report=[report;r]; %#ok<AGROW>
  if isempty(bestModel)||score<best,best=score;bestModel=m;end
 end
 return;
end
% Same split/search per lambda. No test input. Always retain lambda=0 reference.
weights=unique([0 cfg.oeTransitionWeight(:)']);
if any(~isfinite(weights))||any(weights<0),error('Invalid transition weights');end
best=Inf;bestModel=[];report=table();weighted=struct();
for j=1:numel(weights)
 c=cfg;c.oeTransitionWeight=weights(j);
 [m,r,score]=localSearch(tr,va,vDDV,c,lags,betas,ridges);
 if numel(weights)>1
  r.TransitionWeight=repmat(weights(j),height(r),1);
  label=matlab.lang.makeValidName(sprintf('NL_OE_lambda_%g',weights(j)));weighted.(label)=m;
 end
 report=[report;r]; %#ok<AGROW>
 if isempty(bestModel)||score<best,best=score;bestModel=m;end
end
end
function [bestModel,report,best]=localSearch(tr,va,vDDV,cfg,lags,betas,ridges)
% Coarse legacy NL_OE grid, then local refinement ONLY on validation.
c=cfg;c.oeNonlinear=true;c.oeZeroRateAtNull=false;c.oeTanhScale=1;
best=Inf;bestModel=[];rows={};timings=[];dual={};
for beta=betas
 for lag=lags(lags<=128)
  for ridge=ridges
   [m,score,seconds,metric]=localFit(tr,va,vDDV,c,beta,lag,ridge);
   rows(end+1,:)={beta,lag,ridge,score,1,'coarse'};timings(end+1,1)=seconds;dual(end+1,:)={metric.FullFreeRunRMSE,metric.ResetWindowRMSE,metric.HoldDroop,metric.Eligible}; %#ok<AGROW>
   if isempty(bestModel)||(isfinite(score)&&score<best),best=score;bestModel=m;end
  end
 end
 fprintf('NL_OE beta %.5g complete; best validation RMSE %.6g\n',beta,best);
end
if isempty(bestModel),error('No finite nonlinear OE model');end
quick=isfield(cfg,'quick')&&cfg.quick;
if cfg.nlFocus&&cfg.nlRefine&&~quick
 base=bestModel;refined=unique(base.beta*[.5 .75 1 1.25 1.5]);
 if cfg.nlWideBeta,refined=unique([refined base.beta*[2 3]]);end
 if base.beta==0,refined=[0 -.001 -.003];end
 refined=refined(refined<=0&1+refined/cfg.Fs>0);
 for scale=cfg.nlTanhScales
  if ~isfinite(scale)||scale<=0,error('nlTanhScales must be finite and positive');end
  c.oeTanhScale=scale;
  for beta=refined
   if scale==1&&beta==base.beta,continue;end
   [m,score,seconds,metric]=localFit(tr,va,vDDV,c,beta,base.inputLags,base.ridge);
   rows(end+1,:)={beta,base.inputLags,base.ridge,score,scale,'refined'};timings(end+1,1)=seconds;dual(end+1,:)={metric.FullFreeRunRMSE,metric.ResetWindowRMSE,metric.HoldDroop,metric.Eligible}; %#ok<AGROW>
   if isempty(bestModel)||(isfinite(score)&&score<best),best=score;bestModel=m;end
  end
 end
end
if cfg.nlFocus&&cfg.nlJointSearch&&~quick
 base=bestModel;seen=cell2mat(rows(:,[1 2 3 5]));
 grid=bb_nl_joint_grid(base,cfg,seen);
 for j=1:size(grid,1)
  beta=grid(j,1);lag=grid(j,2);ridge=grid(j,3);c.oeTanhScale=grid(j,4);
  [m,score,seconds,metric]=localFit(tr,va,vDDV,c,beta,lag,ridge);
  rows(end+1,:)={beta,lag,ridge,score,c.oeTanhScale,'joint'};timings(end+1,1)=seconds;dual(end+1,:)={metric.FullFreeRunRMSE,metric.ResetWindowRMSE,metric.HoldDroop,metric.Eligible}; %#ok<AGROW>
  if isempty(bestModel)||(isfinite(score)&&score<best),best=score;bestModel=m;end
 end
end
report=cell2table(rows,'VariableNames',{'Beta','Lags','Ridge','ValidationRMSE','TanhScale','Search'});
if cfg.nlDiagnostics||cfg.nlJointSearch,report.TrainSeconds=timings;end
report=[report cell2table(dual,'VariableNames',{'FullFreeRunRMSE','ResetWindowRMSE','HoldDroop','Eligible'})];
report.WindowSeconds=repmat(cfg.oeWindowSeconds,height(report),1);
report.SelectionMode=repmat({cfg.selectionMode},height(report),1);
end
function [model,score,seconds,metric]=localFit(tr,va,vDDV,cfg,beta,lag,ridge)
timer=tic;model=bb_oe_train(tr,cfg,beta,lag,ridge);seconds=toc(timer);model.name='NL_OE';model=bb_ram_bounds(model,tr,cfg);
[score,metric]=bb_ram_score(model,va,vDDV,cfg);
end
