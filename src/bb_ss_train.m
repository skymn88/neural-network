function [model,report,score]=bb_ss_train(base,tr,va,vDDV,cfg)
% Low-order nonlinear state-space residual, fitted on TRAIN full free runs.
% Frozen external-input bank; predicted RAM is the only output feedback.
if isfield(base,'wh')||isfield(base,'stateLeak'),error('SS requires an unmodified base plant');end
values=[];drives=cell(size(tr));
for f=1:numel(tr)
 values=[values;tr(f).S.RAM]; %#ok<AGROW>
 drives{f}=bb_predict_batch(base,tr(f));
 if isfield(base,'rateGain'),drives{f}=drives{f}*base.rateGain;end
 if isfield(base,'rateBias'),drives{f}=drives{f}+base.rateBias;end
end
model=base;model.name='NL_SS';
s=struct('center',mean(values),'scale',max(std(values),1e-9),...
 'driveScale',max(std(vertcat(drives{:})),1e-9),'weights',zeros(1,5));
model.stateSpace=s;
options=optimset('Display','off','MaxFunEvals',cfg.ssMaxEvals,'MaxIter',cfg.ssMaxEvals,'TolX',1e-4,'TolFun',1e-6);
timer=tic;initial=localLoss(zeros(1,5),model,tr,drives,cfg);
[w,loss,exitflag,output]=fminsearch(@(w)localLoss(w,model,tr,drives,cfg),zeros(1,5),options);
if loss>initial||~isfinite(loss),w=zeros(1,5);loss=initial;end
model.stateSpace.weights=w;seconds=toc(timer);[score,metric]=bb_ram_score(model,va,vDDV,cfg);
report=table(sqrt(initial),sqrt(loss),score,seconds,exitflag,output.funcCount,...
 'VariableNames',{'InitialTrainRMSE','TrainRMSE','ValidationRMSE','TrainSeconds','ExitFlag','Evaluations'});
report.FullFreeRunRMSE=metric.FullFreeRunRMSE;report.ResetWindowRMSE=metric.ResetWindowRMSE;report.HoldDroop=metric.HoldDroop;report.Eligible=metric.Eligible;
end
function loss=localLoss(w,m,tr,drives,cfg)
m.stateSpace.weights=w;sse=0;n=0;
for f=1:numel(tr)
 x=tr(f).S.RAM(1);
 for k=2:numel(drives{f})
  x=bb_ram_step(m,x,drives{f}(k));
  if ~isfinite(x)||abs(x-m.stateSpace.center)>20*m.stateSpace.scale,loss=1e100;return;end
  if isfield(m,'ramBounds'),x=min(max(x,m.ramBounds(1)),m.ramBounds(2));end
  if k>=cfg.evalStartIndex,sse=sse+(x-tr(f).S.RAM(k))^2;n=n+1;end
 end
end
loss=sse/max(n,1);
end
