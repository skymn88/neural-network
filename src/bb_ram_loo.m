function report=bb_ram_loo(tr,cfg,rates,ddvLabel)
% Diagnostic fixed-hyperparameter TRAIN LOFO; never changes validation selection.
rows={};labels=fieldnames(rates);
if numel(tr)<3
 warning('bb:LOFOFallback','RAM LOFO needs at least 3 TRAIN flights.');report=table();return;
end
old=rng;cleanup=onCleanup(@()rng(old)); %#ok<NASGU>
[c,kind]=bb_rich_config(cfg,ddvLabel);
for j=1:numel(labels)
 selected=rates.(labels{j});if ~isfield(selected,'objective'),continue;end
 sse=0;n=0;
 for f=1:numel(tr)
  keep=true(1,numel(tr));keep(f)=false;train=tr(keep);
  ddv=bb_fit_candidate(train,c,ddvLabel,{'CoilCurrent'},{'DDV'},kind);
  u=bb_predict_batch(ddv,tr(f));rt=train;
  if strcmp(cfg.ramTrainInput,'oof'),[rt,~]=bb_ddv_oof(train,cfg,ddvLabel);end
  oc=cfg;oc.oeZeroRateAtNull=false;
  if isfield(selected,'oeConfig'),oc=selected.oeConfig;end
  if isfield(selected,'inputBasis'),oc.oeNonlinear=strcmp(selected.inputBasis,'nonlinear');end
  if isfield(selected,'tanhScale'),oc.oeTanhScale=selected.tanhScale;end
  m=bb_oe_train(rt,oc,selected.beta,selected.inputLags,selected.ridge);m=bb_ram_bounds(m,train,cfg);
  p=bb_rollout_rate(m,tr(f),tr(f).S.RAM(1),struct('DDV',u));ix=cfg.evalStartIndex:numel(p);
  sse=sse+sum((p(ix)-tr(f).S.RAM(ix)).^2);n=n+numel(ix);
 end
 rows(end+1,:)={labels{j},sqrt(sse/n),n}; %#ok<AGROW>
end
report=cell2table(rows,'VariableNames',{'Model','LOFO_RAM_RMSE','Samples'});
end
