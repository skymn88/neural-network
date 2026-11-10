function report=bb_ddv_report(models,tr,va,te,cfg)
% Evaluation only; caller selects DDV using validation before requesting test.
sets={tr,va,te};splits={'train','validation','test'};labels=fieldnames(models);rows={};
for j=1:numel(labels)
 for s=1:3
  sse=0;sst=0;n=0;
  for f=1:numel(sets{s})
   fl=sets{s}(f);p=bb_predict_batch(models.(labels{j}),fl);ix=cfg.evalStartIndex:numel(p);y=fl.S.DDV(ix);
   sse=sse+sum((p(ix)-y).^2);sst=sst+sum((y-mean(y)).^2);n=n+numel(ix);
  end
  fit=NaN;if sst>1e-14,fit=100*(1-sqrt(sse/sst));end
  rows(end+1,:)={labels{j},'DDV',splits{s},fit,sqrt(sse/n)}; %#ok<AGROW>
 end
end
report=cell2table(rows,'VariableNames',{'Model','Stage','Split','FitPct','RMSE'});
end
