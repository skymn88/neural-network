function report=bb_alignment_report(tr,va,te,cfg)
% Positive lag: response follows input. Correlation is NOT a delay correction.
sets={tr,va,te};splits={'train','validation','test'};rows={};
for s=1:3
 for f=1:numel(sets{s})
  fl=sets{s}(f);inputs={fl.S.CoilCurrent,fl.S.DDV};
  outputs={fl.S.DDV,[0;diff(fl.S.RAM)]*cfg.Fs};pairs={'Current_DDV','DDV_RAMRate'};
  for j=1:2
   u=inputs{j};y=outputs{j};limit=min(cfg.alignmentMaxLag,numel(u)-3);
   lags=-limit:limit;score=nan(size(lags));
   for k=1:numel(lags)
    lag=lags(k);ix=max(1,1-lag):min(numel(u),numel(u)-lag);
    a=u(ix);b=y(ix+lag);a=a-mean(a);b=b-mean(b);den=norm(a)*norm(b);
    if den>1e-12,score(k)=(a'*b)/den;end
   end
   valid=find(isfinite(score));delay=NaN;peak=NaN;
   if ~isempty(valid),[~,i]=max(abs(score(valid)));i=valid(i);delay=lags(i);peak=score(i);end
   note='Diagnostic only; causal FIR cannot represent a negative delay; periodic inputs can be ambiguous.';
   rows(end+1,:)={fl.name,splits{s},pairs{j},delay,peak,cfg.Fs,note}; %#ok<AGROW>
  end
 end
end
report=cell2table(rows,'VariableNames',{'Flight','Split','Pair','LagSamples','Correlation','Fs','Note'});
end
