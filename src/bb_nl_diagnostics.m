function [curve,holds,equilibrium]=bb_nl_diagnostics(report,plant,tr,va,te,cfg)
% Profile curve is min over OTHER settings, not a fixed-hyperparameter curve.
betas=unique(report.Beta);rows={};coarse=report(strcmp(report.Search,'coarse'),:);
[~,winner]=min(report.ValidationRMSE);win=report.Beta(winner);
[~,ci]=min(coarse.ValidationRMSE);cw=coarse.Beta(ci);
edge=win==min(betas)||win==max(betas);coarseEdge=cw==min(coarse.Beta)||cw==max(coarse.Beta);
if edge||coarseEdge,warning('bb:BetaEdge','Validation beta is at a searched grid edge; inspect BetaCurve, not test scores.');end
for j=1:numel(betas)
 q=report(report.Beta==betas(j),:);[score,k]=min(q.ValidationRMSE);
 rows(end+1,:)={betas(j),score,q.Lags(k),q.Ridge(k),q.TanhScale(k),height(q),betas(j)==win,coarseEdge,edge}; %#ok<AGROW>
end
curve=cell2table(rows,'VariableNames',{'Beta','BestValidationRMSE','Lags','Ridge','TanhScale','Trials','BestBeta','CoarseEdge','SearchedEdge'});
x=[];for f=1:numel(tr),x=[x;tr(f).S.RAM];end %#ok<AGROW>
threshold=max(1e-12,.01*(max(x)-min(x)));rows=cell(0,10);sets={tr,va,te};splits={'train','validation','test'};
for s=1:3
 for f=1:numel(sets{s})
  fl=sets{s}(f);u=bb_predict_batch(plant.ddv,fl);p=bb_rollout_rate(plant.rate,fl,fl.S.RAM(1),struct('DDV',u));
  moving=conv(double([false;abs(diff(fl.S.RAM))>threshold]),ones(2*cfg.Fs+1,1));
  mask=moving(1:numel(p))==0;mask(1:cfg.evalStartIndex-1)=false;
  edges=diff([false;mask;false]);a=find(edges==1);b=find(edges==-1)-1;
  for k=1:numel(a)
   ix=a(k):b(k);if numel(ix)<2,continue;end
   ps=std(p(ix));ys=std(fl.S.RAM(ix));ratio=NaN;if ys>1e-12,ratio=ps/ys;end
   rows(end+1,:)={fl.name,splits{s},k,numel(ix),ps,ys,ratio,mean(p(ix)),mean(fl.S.RAM(ix)),mean(x)}; %#ok<AGROW>
  end
 end
end
holds=cell2table(rows,'VariableNames',{'Flight','Split','Segment','Samples','PredictionStd','MeasuredStd','StdRatio','PredictionMean','MeasuredMean','TrainMean'});
m=plant.rate;null.t=(0:max(m.inputLags,8))'/cfg.Fs;names=m.inputNames;
if isfield(m,'featureSpec')
 if m.featureSpec.force,names=[names {'ActuatorForce'}];end
 if m.featureSpec.pressure,names=[names {'SupplyPressure','ReturnPressure'}];end
end
for j=1:numel(names)
 value=0;
 if ~strcmp(names{j},'DDV')
  values=[];for f=1:numel(tr),values=[values;tr(f).S.(names{j})];end %#ok<AGROW>
  value=mean(values);
 end
 null.S.(names{j})=repmat(value,numel(null.t),1);
end
if isfield(m,'filterBank')
 u=zeros(1,numel(m.inputNames));
 for j=1:numel(m.inputNames),u(j)=null.S.(m.inputNames{j})(1);end
 sections=m.filterBank.rawLags+1+numel(m.filterBank.poles)+size(m.filterBank.a,1);
 v=bb_forward(m,repmat(u,1,sections)); % unity DC gain, exact steady-state bank features
else,v=bb_predict_batch(m,null);end
drive=v(end)*m.rateGain+m.rateBias;
if isfield(m,'wh'),drive=bb_wh_map(m.wh,drive);end
tau=Inf;if m.beta<0,tau=-1/(cfg.Fs*log(1+m.beta/cfg.Fs));end
xeq=NaN;stable=false;note='Unclamped equilibrium, DDV history zero; auxiliary inputs at TRAIN means.';
if m.beta<0
 xeq=-drive/m.beta;
 if isfield(m,'stateLeak')
  q=m.stateLeak;radius=(abs(drive)+q.k)/abs(m.beta)+abs(q.center)+1;
  xeq=fzero(@(z)m.beta*z+drive-q.k*tanh((z-q.center)/q.scale),[-radius radius]);
 end
 derivative=1+m.beta/cfg.Fs;
 if isfield(m,'stateLeak'),derivative=derivative-q.k/(cfg.Fs*q.scale)*(1-tanh((xeq-q.center)/q.scale)^2);end
 stable=abs(derivative)<1;
else
 note=[note ' beta=0: no unique linear equilibrium.'];
end
equilibrium=table(m.beta,tau,drive,xeq,stable,mean(x),min(x),max(x),{note},...
 'VariableNames',{'Beta','LinearMemorySeconds','ZeroDDVRate','EquilibriumRAM','LocallyStable','TrainMean','TrainMin','TrainMax','Note'});
end
