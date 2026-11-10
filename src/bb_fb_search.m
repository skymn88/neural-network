function [model,report,best]=bb_fb_search(tr,va,vDDV,cfg,betas,ridges)
% Fixed bank; only beta/ridge/lambda selected on validation, no test input.
best=Inf;model=[];rows={};c=cfg;c.oeFilterBank=true;c.oeZeroRateAtNull=false;
for weight=unique([0 cfg.oeTransitionWeight(:)'])
 c.oeTransitionWeight=weight;
 for beta=betas
  for ridge=ridges
   fitTimer=tic;m=bb_oe_train(tr,c,beta,cfg.fbRawLags,ridge);seconds=toc(fitTimer);m=bb_ram_bounds(m,tr,cfg);
   sse=0;n=0;
   for f=1:numel(va)
    p=bb_rollout_rate(m,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=cfg.evalStartIndex:numel(p);
    sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
   end
   score=sqrt(sse/n);rows(end+1,:)={beta,ridge,weight,numel(m.weights),seconds,score}; %#ok<AGROW>
   if isfinite(score)&&score<best,best=score;model=m;model.name='NL_FB';end
  end
 end
end
if isempty(model),error('No finite NL_FB validation model');end
report=cell2table(rows,'VariableNames',{'Beta','Ridge','TransitionWeight','Coefficients','TrainSeconds','ValidationRMSE'});
end
