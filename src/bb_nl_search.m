function [bestModel,report,best]=bb_nl_search(tr,va,vDDV,cfg,lags,betas,ridges)
% Coarse legacy NL_OE grid, then local refinement ONLY on validation.
c=cfg;c.oeNonlinear=true;c.oeZeroRateAtNull=false;c.oeTanhScale=1;
best=Inf;bestModel=[];rows={};
for beta=betas
 for lag=lags(lags<=128)
  for ridge=ridges
   [m,score]=localFit(tr,va,vDDV,c,beta,lag,ridge);
   rows(end+1,:)={beta,lag,ridge,score,1,'coarse'}; %#ok<AGROW>
   if isfinite(score)&&score<best,best=score;bestModel=m;end
  end
 end
 fprintf('NL_OE beta %.5g complete; best validation RMSE %.6g\n',beta,best);
end
if isempty(bestModel),error('No finite nonlinear OE model');end
quick=isfield(cfg,'quick')&&cfg.quick;
if cfg.nlFocus&&cfg.nlRefine&&~quick
 base=bestModel;refined=unique(base.beta*[.5 .75 1 1.25 1.5]);
 if base.beta==0,refined=[0 -.001 -.003];end
 refined=refined(refined<=0&1+refined/cfg.Fs>0);
 for scale=cfg.nlTanhScales
  if ~isfinite(scale)||scale<=0,error('nlTanhScales must be finite and positive');end
  c.oeTanhScale=scale;
  for beta=refined
   if scale==1&&beta==base.beta,continue;end
   [m,score]=localFit(tr,va,vDDV,c,beta,base.inputLags,base.ridge);
   rows(end+1,:)={beta,base.inputLags,base.ridge,score,scale,'refined'}; %#ok<AGROW>
   if isfinite(score)&&score<best,best=score;bestModel=m;end
  end
 end
end
report=cell2table(rows,'VariableNames',{'Beta','Lags','Ridge','ValidationRMSE','TanhScale','Search'});
end
function [model,score]=localFit(tr,va,vDDV,cfg,beta,lag,ridge)
model=bb_oe_train(tr,cfg,beta,lag,ridge);model.name='NL_OE';model=bb_ram_bounds(model,tr,cfg);sse=0;n=0;
for f=1:numel(va)
 p=bb_rollout_rate(model,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=cfg.evalStartIndex:numel(p);
 sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
end
score=sqrt(sse/n);
end
