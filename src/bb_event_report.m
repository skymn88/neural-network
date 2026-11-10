function report=bb_event_report(configs,tr,va,te,cfg)
% Diagnostic masks only; never changes training or model selection.
% Transition: |delta RAM| > 1% TRAIN RAM range, plus 2s following each event.
x=[];for f=1:numel(tr),x=[x;tr(f).S.RAM];end %#ok<AGROW>
threshold=max(1e-12,.01*(max(x)-min(x)));
weighted=any(cfg.oeTransitionWeight~=0);
if weighted,threshold=max(1e-12,cfg.oeTransitionFraction*(max(x)-min(x)));end
sets={tr,va,te};names={'train','validation','test'};
labels=fieldnames(configs);rows={};drifts=[];
for s=1:3
 fs=sets{s};
 for f=1:numel(fs)
  fl=fs(f);N=numel(fl.t);event=[false;abs(diff(fl.S.RAM))>threshold];
  expanded=conv(double(event),ones(2*cfg.Fs+1,1),'full');moving=expanded(1:N)>0;
  if weighted,moving=bb_transition_mask(fl.S.RAM,cfg.Fs,threshold,cfg.oeTransitionSeconds);end
  eligible=false(N,1);eligible(cfg.evalStartIndex:end)=true;
  for j=1:numel(labels)
   c=configs.(labels{j});u=bb_predict_batch(c.ddv,fl);p=bb_rollout_rate(c.rate,fl,fl.S.RAM(1),struct('DDV',u));
   for q=1:2
    if q==1,mask=moving&eligible;phase='transition';else,mask=~moving&eligible;phase='hold';end
    n=sum(mask);rmse=NaN;bias=NaN;maxError=NaN;
    if n>0,e=p(mask)-fl.S.RAM(mask);rmse=sqrt(mean(e.^2));bias=mean(e);maxError=max(abs(e));end
    if cfg.diagnostics
        slope=NaN;worst=NaN;count=0;
        if q==2,[slope,worst,count]=bb_hold_drift(p-fl.S.RAM,mask,cfg.Fs);end
        drifts(end+1,:)=[slope worst count]; %#ok<AGROW>
    end
    rows(end+1,:)={labels{j},names{s},fl.name,phase,n,rmse,bias,maxError,threshold}; %#ok<AGROW>
   end
  end
 end
end
report=cell2table(rows,'VariableNames',{'Model','Split','Flight','Phase','Samples','RAM_RMSE','RAM_Bias','RAM_MaxAbs','DetectionThreshold'});
if weighted
 report.TransitionWeight=nan(height(report),1);
 for j=1:numel(labels)
  m=configs.(labels{j}).rate;
  if isfield(m,'transitionWeight'),report.TransitionWeight(strcmp(report.Model,labels{j}))=m.transitionWeight;end
 end
end
if cfg.diagnostics
    report.HoldErrorSlope=drifts(:,1);report.HoldWorstAbsSlope=drifts(:,2);report.HoldSegments=drifts(:,3);
end
end
