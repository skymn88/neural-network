function [rateMap,steps]=bb_rate_diagnostics(plant,tr,va,cfg)
% Measured-DDV component diagnostic. TRAIN-derived bins/threshold; no test input.
B=cfg.rateMapBins;if B<2||B~=floor(B),error('rateMapBins must be integer >=2');end
x=[];speed=[];
for f=1:numel(tr)
 x=[x;tr(f).S.RAM];v=[0;diff(tr(f).S.RAM)]*cfg.Fs;speed=[speed;v(cfg.evalStartIndex:end)]; %#ok<AGROW>
end
lo=min(speed);hi=max(speed);if hi-lo<1e-12,lo=lo-1;hi=hi+1;end
edges=linspace(lo,hi,B+1);edges(1)=-Inf;edges(end)=Inf;
threshold=max(1e-12,cfg.oeTransitionFraction*(max(x)-min(x)));
sets={tr,va};names={'train','validation'};rows=cell(0,8);events=cell(0,12);
for s=1:2
 count=zeros(B,1);sy=count;sp=count;sr=count;
 for f=1:numel(sets{s})
  fl=sets{s}(f);[p,r]=bb_rollout_rate(plant.rate,fl,fl.S.RAM(1));
  if isfield(plant.rate,'dynamicMLP'),r=[0;diff(p)]*cfg.Fs;end
  v=[0;diff(fl.S.RAM)]*cfg.Fs;vp=[0;diff(p)]*cfg.Fs;ix=cfg.evalStartIndex:numel(v);
  for b=1:B
   mask=v(ix)>=edges(b)&v(ix)<edges(b+1);ii=ix(mask);
   count(b)=count(b)+numel(ii);sy(b)=sy(b)+sum(v(ii));sp(b)=sp(b)+sum(vp(ii));sr(b)=sr(b)+sum(r(ii));
  end
  if s==2
   moving=bb_transition_mask(fl.S.RAM,cfg.Fs,threshold,cfg.oeTransitionSeconds);
   moving(1:cfg.evalStartIndex-1)=false;ends=diff([false;moving;false]);a=find(ends==1);z=find(ends==-1)-1;
   for j=1:numel(a)
    first=max(1,a(j)-1);last=z(j);ii=first:last;start=fl.S.RAM(first);
    [amp,k]=max(abs(fl.S.RAM(ii)-start));if amp<=1e-12,continue;end
    target=fl.S.RAM(ii(k));direction=sign(target-start);
    dm=localDuration(fl.S.RAM(ii),start,amp,direction,cfg.Fs);
    dp=localDuration(p(ii),start,amp,direction,cfg.Fs);
    pm=max(abs(v(a(j):last)));pp=max(abs(vp(a(j):last)));
    peakRatio=NaN;if pm>1e-12,peakRatio=pp/pm;end
    durationRatio=NaN;if isfinite(dm)&&dm>0,durationRatio=dp/dm;end
    events(end+1,:)={fl.name,fl.t(first),fl.t(last),amp,pm,pp,peakRatio,dm,dp,durationRatio,isfinite(dp),'measured DDV'}; %#ok<AGROW>
   end
  end
 end
 for b=1:B
  ym=NaN;yp=NaN;yr=NaN;
  if count(b)>0,ym=sy(b)/count(b);yp=sp(b)/count(b);yr=sr(b)/count(b);end
  rows(end+1,:)={names{s},b,edges(b),edges(b+1),count(b),ym,yp,yr}; %#ok<AGROW>
 end
end
rateMap=cell2table(rows,'VariableNames',{'Split','Bin','LowerMeasuredRate','UpperMeasuredRate','Count','MeanMeasuredRate','MeanPredictedRAMRate','MeanModelDrive'});
steps=cell2table(events,'VariableNames',{'Flight','StartSeconds','EndSeconds','MeasuredAmplitude','MeasuredPeakRate','PredictedPeakRate','PredOverMeasuredPeak','MeasuredDuration10to90','PredictedDuration10to90','PredOverMeasuredDuration','CrossedBothLevels','Input'});
steps.MeasuredOverPredPeak=nan(height(steps),1);
steps.MeasuredOverPredDuration=nan(height(steps),1);
valid=isfinite(steps.PredictedPeakRate)&steps.PredictedPeakRate>0;
steps.MeasuredOverPredPeak(valid)=steps.MeasuredPeakRate(valid)./steps.PredictedPeakRate(valid);
valid=isfinite(steps.PredictedDuration10to90)&steps.PredictedDuration10to90>0;
steps.MeasuredOverPredDuration(valid)=steps.MeasuredDuration10to90(valid)./steps.PredictedDuration10to90(valid);
if ~isempty(steps),[~,order]=sort(steps.MeasuredAmplitude,'descend');steps=steps(order(1:min(5,numel(order))),:);end
end
function seconds=localDuration(y,start,amplitude,direction,Fs)
progress=direction*(y-start)/amplitude;i=find(progress>=.1,1);j=find(progress>=.9,1);
seconds=NaN;if ~isempty(i)&&~isempty(j)&&j>=i,seconds=(j-i)/Fs;end
end
