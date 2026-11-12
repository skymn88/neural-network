function [candidate,report,bestScore]=bb_state_leak_fit(base,tr,va,vDDV,cfg)
% Fit one bounded leakage strength on TRAIN free runs for each fixed scale.
% Validation can reject every nonlinear candidate; original remains eligible.
x=[];rates=cell(size(tr));
for f=1:numel(tr)
 x=[x;tr(f).S.RAM];rates{f}=bb_predict_batch(base,tr(f))*base.rateGain+base.rateBias; %#ok<AGROW>
end
spread=std(x);if spread<1e-12,error('State leakage needs TRAIN RAM excitation');end
if cfg.leakMaxRate<=0||cfg.leakMaxEvals<1,error('Invalid leakage search budget');end
candidate=base;bestScore=localValidation(base,va,vDDV,cfg);rows={};
rows(end+1,:)={0,NaN,mean(x),localLoss(base,tr,rates,cfg),bestScore,0,true};
for factor=cfg.leakScaleFactors
 if factor<=0||~isfinite(factor),error('Invalid leakage scale');end
 m=base;m.stateLeak=struct('k',0,'scale',spread*factor,'center',mean(x));
 upper=m.stateLeak.scale*min(cfg.leakMaxRate,.95*(cfg.Fs+base.beta));
 if upper<=0,continue;end
 options=optimset('Display','off','MaxFunEvals',cfg.leakMaxEvals,'TolX',max(1e-12,upper*1e-4));
 timer=tic;[k,~]=fminbnd(@(k)localObjective(k,m,tr,rates,cfg),0,upper,options);
 m.stateLeak.k=k;trainScore=localLoss(m,tr,rates,cfg);seconds=toc(timer);
 score=localValidation(m,va,vDDV,cfg);
 rows(end+1,:)={k,m.stateLeak.scale,m.stateLeak.center,trainScore,score,seconds,false}; %#ok<AGROW>
 if isfinite(score)&&score<bestScore,bestScore=score;candidate=m;end
end
report=cell2table(rows,'VariableNames',{'LeakK','LeakScale','LeakCenter','TrainRMSE','ValidationRMSE','TrainSeconds','Original'});
end
function score=localObjective(k,m,tr,rates,cfg)
m.stateLeak.k=k;score=localLoss(m,tr,rates,cfg);
end
function score=localLoss(m,tr,rates,cfg)
sse=0;n=0;
for f=1:numel(tr)
 state=tr(f).S.RAM(1);
 if isfield(m,'ramBounds'),state=min(max(state,m.ramBounds(1)),m.ramBounds(2));end
 for k=2:numel(tr(f).t)
  state=bb_ram_step(m,state,rates{f}(k));
  if isfield(m,'ramBounds'),state=min(max(state,m.ramBounds(1)),m.ramBounds(2));end
  if k>=cfg.evalStartIndex,sse=sse+(state-tr(f).S.RAM(k))^2;n=n+1;end
 end
end
score=sqrt(sse/n);if ~isfinite(score),score=realmax;end
end
function score=localValidation(m,va,vDDV,cfg)
sse=0;n=0;
for f=1:numel(va)
 p=bb_rollout_rate(m,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=cfg.evalStartIndex:numel(p);
 sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
end
score=sqrt(sse/n);if ~isfinite(score),score=Inf;end
end
