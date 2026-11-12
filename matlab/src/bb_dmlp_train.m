function [m,report,score]=bb_dmlp_train(tr,va,vDDV,cfg)
% Small network, all weights trained jointly on multi-step TRAIN windows.
% Validation is full-flight free run; no test input or teacher forcing there.
if cfg.useForce||cfg.usePressure,error('Dynamic MLP currently requires DDV-only RAM inputs');end
old=rng;cleanup=onCleanup(@()rng(old));rng(cfg.seed); %#ok<NASGU>
H=cfg.dmlpHidden;T=cfg.dmlpHorizon;
if cfg.dmlpHoldFraction<=0||cfg.dmlpHoldFraction>1||cfg.dmlpRateSmoothSeconds<0,error('Invalid movement/hold settings');end
if H<1||H~=floor(H)||T<2||T~=floor(T)||cfg.dmlpWindows<numel(tr),error('Invalid dynamic MLP dimensions/budget');end
m=struct('name','DMLP','kind','DMLP','Fs',cfg.Fs,'inputNames',{{'DDV'}},...
 'outputNames',{{'RAMRate'}},'inputLags',8,'outputLags',0,'beta',0,...
 'rateGain',1,'rateBias',0,'nInputs',1,'nOutputs',1);
if cfg.nullCenter
 m.ddvCenter=cfg.nullDDVCenter;
 for f=1:numel(tr),tr(f).S.DDV=tr(f).S.DDV-cfg.nullDDVCenter;end
end
inputs=cell(size(tr));rates=cell(size(tr));X=[];Y=[];positions=[];
for f=1:numel(tr)
 lag=bb_lag_inputs(tr(f),{'DDV'},8);inputs{f}=bb_dmlp_inputs(lag);
 rates{f}=[0;diff(tr(f).S.RAM)]*cfg.Fs;first=max(2,cfg.evalStartIndex);last=numel(tr(f).t);
 if cfg.dmlpBalancedLoss
  width=max(1,round(cfg.dmlpRateSmoothSeconds*cfg.Fs));rates{f}=conv(rates{f},ones(width,1)/width,'same');
  first=max(first,width+1);last=last-width;
 end
 ix=first:last;if isempty(ix),error('TRAIN too short for velocity target smoothing');end
 X=[X;inputs{f}(ix,:)];positions=[positions;tr(f).S.RAM(ix-1)]; %#ok<AGROW>
 Y=[Y;rates{f}(ix)]; %#ok<AGROW>
end
mu=mean(X,1);sigma=std(X,0,1);sigma(sigma<1e-10)=1;
m.xMean=zeros(1,9);m.xStd=ones(1,9);m.yMean=0;m.yStd=1;
q=struct('uMean',mu,'uStd',sigma,'xCenter',mean(positions),...
 'xScale',max(std(positions),1e-9),'rateScale',max(std(Y),1e-9),...
 'W',.3*randn(H,4),'b',.1*randn(H,1),'v',zeros(H,1),'bias',0);
Z=[bsxfun(@rdivide,bsxfun(@minus,X,mu),sigma) (positions-q.xCenter)/q.xScale];
A=[tanh(bsxfun(@plus,Z*q.W',q.b')) ones(size(Z,1),1)];
w=(A'*A/size(A,1)+1e-3*eye(H+1))\(A'*(Y/q.rateScale)/size(A,1));q.v=w(1:H);q.bias=w(end);m.dynamicMLP=q;m=bb_ram_bounds(m,tr,cfg);
% Coverage and (when enabled) movement windows spanning every TRAIN flight. All batches advanced together.
sorted=sort(abs(Y));holdThreshold=sorted(max(1,ceil(cfg.dmlpHoldFraction*numel(Y))));
U=zeros(T,0,3);target=zeros(T,0);targetRate=zeros(T,0);x0=[];per=max(1,floor(cfg.dmlpWindows/numel(tr)));
for f=1:numel(tr)
 first=max(2,cfg.evalStartIndex);last=numel(tr(f).t)-T+1;
 if cfg.dmlpBalancedLoss,first=max(first,width+1);last=last-width;end
 if last<first,error('TRAIN flight too short for dynamic MLP horizon');end
 starts=unique(round(linspace(first,last,min(per,last-first+1))));
 if cfg.dmlpBalancedLoss,starts=localStarts(first,last,per,T,rates{f});end
 for start=starts
  ix=start:start+T-1;k=size(target,2)+1;
  U(:,k,:)=reshape(bsxfun(@rdivide,bsxfun(@minus,inputs{f}(ix,:),mu),sigma),T,1,3);
  target(:,k)=tr(f).S.RAM(ix);targetRate(:,k)=rates{f}(ix);x0(k,1)=tr(f).S.RAM(start-1); %#ok<AGROW>
 end
end
p=[q.W(:);q.b;q.v;q.bias];[initial,initialParts]=localLoss(p,m,U,target,targetRate,x0,holdThreshold,cfg);
options=optimset('Display','off','MaxFunEvals',cfg.dmlpMaxEvals,'MaxIter',cfg.dmlpMaxEvals,'TolX',1e-4,'TolFun',1e-7);
timer=tic;[best,loss,flag,output]=fminsearch(@(p)localLoss(p,m,U,target,targetRate,x0,holdThreshold,cfg),p,options);
if ~isfinite(loss)||loss>initial,best=p;loss=initial;end
[loss,parts]=localLoss(best,m,U,target,targetRate,x0,holdThreshold,cfg);
m=localUnpack(best,m);seconds=toc(timer);m=bb_ram_bounds(m,tr,cfg);
[score,metric]=bb_ram_score(m,va,vDDV,cfg);
report=table(H,T,size(target,2),sqrt(initialParts(1))*q.xScale,sqrt(parts(1))*q.xScale,score,seconds,flag,output.funcCount,...
 'VariableNames',{'Hidden','Horizon','Windows','InitialWindowRMSE','TrainWindowRMSE','ValidationRMSE','TrainSeconds','ExitFlag','Evaluations'});
report.InitialObjective=initial;report.TrainObjective=loss;
report.MoveRateRMSE=sqrt(parts(2))*q.rateScale;report.HoldRateRMSE=sqrt(parts(3))*q.rateScale;
report.DisplacementRMSE=sqrt(parts(4))*q.xScale;report.HoldSpeedThreshold=holdThreshold;
report.HoldSamples=sum(abs(targetRate(:))<=holdThreshold);report.MoveSamples=numel(targetRate)-report.HoldSamples;
report.BalancedLoss=cfg.dmlpBalancedLoss;
report.FullFreeRunRMSE=metric.FullFreeRunRMSE;report.ResetWindowRMSE=metric.ResetWindowRMSE;report.HoldDroop=metric.HoldDroop;report.Eligible=metric.Eligible;
end
function m=localUnpack(p,m)
H=numel(m.dynamicMLP.b);q=m.dynamicMLP;q.W=reshape(p(1:4*H),H,4);q.b=p(4*H+(1:H));q.v=p(5*H+(1:H));q.bias=p(end);m.dynamicMLP=q;
end
function [loss,parts]=localLoss(p,m,U,target,targetRate,x,threshold,cfg)
m=localUnpack(p,m);sse=0;move=0;holdError=0;nm=0;nh=0;
for k=1:size(U,1)
 u=reshape(U(k,:,:),size(U,2),3);previous=x;x=bb_dmlp_step(m,x,u);
 if isfield(m,'ramBounds'),x=min(max(x,m.ramBounds(1)),m.ramBounds(2));end
 if any(~isfinite(x))||any(abs(x-m.dynamicMLP.xCenter)>20*m.dynamicMLP.xScale)
  loss=1e100;parts=repmat(1e100,1,4);return;
 end
 sse=sse+sum(((x-target(k,:)')/m.dynamicMLP.xScale).^2);
 e=((x-previous)*m.Fs-targetRate(k,:)')/m.dynamicMLP.rateScale;
 mask=abs(targetRate(k,:)')<=threshold;
 holdError=holdError+sum(e(mask).^2);nh=nh+sum(mask);
 move=move+sum(e(~mask).^2);nm=nm+sum(~mask);
end
parts=[sse/numel(target),move/max(nm,1),holdError/max(nh,1),mean(((x-target(end,:)')/m.dynamicMLP.xScale).^2)];
loss=parts(1);
if cfg.dmlpBalancedLoss
 weights=[cfg.dmlpPositionWeight cfg.dmlpMoveRateWeight cfg.dmlpHoldRateWeight cfg.dmlpDisplacementWeight];
 if any(~isfinite(weights))||any(weights<0)||sum(weights)==0,error('Invalid dynamic MLP loss weights');end
 loss=parts*weights';
end
end
function starts=localStarts(first,last,budget,T,rate)
% TRAIN-only mix of coverage and large-motion windows, deterministic.
budget=min(budget,last-first+1);starts=unique(round(linspace(first,last,max(1,ceil(budget/2)))));
[~,order]=sort(abs(rate(first:last)),'descend');
for k=1:numel(order)
 if numel(starts)>=budget,break;end
 candidate=max(first,min(last,first+order(k)-1-floor(T/4)));
 if all(abs(starts-candidate)>=max(1,floor(T/2))),starts(end+1)=candidate;end %#ok<AGROW>
 if numel(starts)>=budget,break;end
end
if numel(starts)<budget
 remaining=setdiff(first:last,starts);take=unique(round(linspace(1,numel(remaining),budget-numel(starts))));
 starts=[starts remaining(take)];
end
starts=sort(starts);
end
