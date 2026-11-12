function [summary,report]=bb_closed_report(closed,tr,va,te,cfg)
sets={tr,va,te};splits={'train','validation','test'};labels={'SP_OE','Hybrid'};rows={};sr={};
y=[];for f=1:numel(tr),y=[y;tr(f).S.RAM];end %#ok<AGROW>
scale=std(y);threshold=max(1e-12,.01*(max(y)-min(y)));
for s=1:3
 fs=sets{s};
 for j=1:2
  total=0;sse=0;sst=0;maxerr=0;bias=0;covered=0;failed=false;
  for f=1:numel(fs)
   fl=fs(f);
   if j==1
    [p,~,info]=bb_rollout_rate(closed.setpoint,fl,fl.S.RAM(1));
    divergence=info.diverged||any(~isfinite(p))||any(abs(p)>max(eps,2*max(abs(closed.plant.trainBounds))));
   else
    [p,info]=bb_closed_loop(closed.controller,closed.plant,fl,fl.S.RAM(1));divergence=info.diverged;
   end
   ix=cfg.evalStartIndex:numel(p);e=p(ix)-fl.S.RAM(ix);truth=fl.S.RAM(ix);valid=isfinite(e);
   failed=failed||any(~valid);total=total+numel(e);covered=covered+sum(valid);
   sse=sse+sum(e.^2);sst=sst+sum((truth-mean(truth)).^2);bias=bias+sum(e);maxerr=max(maxerr,max(abs(e)));
   event=[false;abs(diff(fl.S.RAM))>threshold];moving=conv(double(event),ones(2*cfg.Fs+1,1));
   hold=moving(1:numel(p))==0;hold(1:cfg.evalStartIndex-1)=false;
   [slope,worst,count]=bb_hold_drift(p-fl.S.RAM,hold,cfg.Fs);
   rows(end+1,:)={labels{j},splits{s},fl.name,sqrt(mean(e.^2)),divergence,slope,worst,count}; %#ok<AGROW>
  end
  fit=NaN;if sst>1e-14,fit=100*(1-sqrt(sse/sst));end
  rmse=sqrt(sse/total);nr=NaN;if scale>1e-12,nr=100*rmse/scale;end
  if failed,fit=NaN;rmse=Inf;nr=Inf;maxerr=Inf;end
  sr(end+1,:)={labels{j},'RAM_Closed',splits{s},fit,NaN,rmse,nr,100*covered/total,maxerr,bias/total}; %#ok<AGROW>
 end
end
summary=cell2table(sr,'VariableNames',{'Model','Stage','Split','FitPct','OneStepFitPct','RMSE','TrainStdNRMSEpct','CoveragePct','MaxAbsError','MeanBias'});
report=cell2table(rows,'VariableNames',{'Model','Split','Flight','RAM_RMSE','Diverged','HoldErrorSlope','HoldWorstAbsSlope','HoldSegments'});
end
