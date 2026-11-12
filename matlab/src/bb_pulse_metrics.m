function [report,droop]=bb_pulse_metrics(fl,y,Fs)
% Per-phase slope/level drift; transition times relative to achieved pulse change.
rows={};amplitudes=[];commands=[];droop=0;
for j=1:size(fl.pulseProfile,1)
 ix=find(fl.pulseSegment==j);first=max(1,ix(1)-1);last=ix(end);
 initial=y(first);delta=y(last)-initial;slope=NaN;pct=NaN;pctRate=NaN;duration=NaN;
 if fl.pulseProfile(j,1)==0
  if numel(ix)>1
   t=(0:numel(ix)-1)'/Fs;t=t-mean(t);slope=sum(t.*(y(ix)-mean(y(ix))))/sum(t.^2);
   droop=max(droop,abs(slope));
  end
  if abs(initial)>1e-12,pct=100*(y(last)-initial)/abs(initial);pctRate=100*slope/abs(initial);end
  phase='hold';
 else
  if abs(delta)>1e-12
   progress=(y(first:last)-initial)/delta;a=find(progress>=.1,1);b=find(progress>=.9,1);
   if ~isempty(a)&&~isempty(b)&&b>=a,duration=(b-a)/Fs;end
  end
  phase='pulse';amplitudes(end+1)=delta;commands(end+1)=fl.pulseProfile(j,1)*fl.pulseProfile(j,2); %#ok<AGROW>
 end
 rows(end+1,:)={j,phase,initial,y(last),delta,slope,pct,pctRate,duration}; %#ok<AGROW>
end
report=cell2table(rows,'VariableNames',{'Segment','Phase','StartLevel','EndLevel','Change','DriftPerSecond','DriftPctOfPulseEnd','DriftPctPerSecond','Duration10to90'});
report.CurrentAmplitude=zeros(height(report),1);report.DurationSeconds=zeros(height(report),1);
for j=1:height(report)
 ix=find(fl.pulseSegment==j);report.CurrentAmplitude(j)=fl.S.CoilCurrent(ix(1));report.DurationSeconds(j)=numel(ix)/Fs;
end
symmetry=NaN;pos=find(commands>0,1);neg=find(commands<0,1);
if ~isempty(pos)&&~isempty(neg)
 gp=amplitudes(pos)/commands(pos);gn=amplitudes(neg)/commands(neg);
 if abs(gp)>1e-12,symmetry=gn/gp;end
end
report.NegOverPosAreaNormalizedGain=repmat(symmetry,height(report),1);
if any(~isfinite(y)),droop=Inf;end
end
