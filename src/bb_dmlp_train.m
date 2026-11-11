function [m,report,score]=bb_dmlp_train(tr,va,vDDV,cfg)
% Small network, all weights trained jointly on multi-step TRAIN windows.
% Validation is full-flight free run; no test input or teacher forcing there.
if cfg.useForce||cfg.usePressure,error('Dynamic MLP currently requires DDV-only RAM inputs');end
old=rng;cleanup=onCleanup(@()rng(old));rng(cfg.seed); %#ok<NASGU>
H=cfg.dmlpHidden;T=cfg.dmlpHorizon;
if H<1||H~=floor(H)||T<2||T~=floor(T)||cfg.dmlpWindows<numel(tr),error('Invalid dynamic MLP dimensions/budget');end
m=struct('name','DMLP','kind','DMLP','Fs',cfg.Fs,'inputNames',{{'DDV'}},...
 'outputNames',{{'RAMRate'}},'inputLags',8,'outputLags',0,'beta',0,...
 'rateGain',1,'rateBias',0,'nInputs',1,'nOutputs',1);
inputs=cell(size(tr));X=[];Y=[];positions=[];
for f=1:numel(tr)
 lag=bb_lag_inputs(tr(f),{'DDV'},8);inputs{f}=bb_dmlp_inputs(lag);
 ix=max(2,cfg.evalStartIndex):numel(lag(:,1));
 X=[X;inputs{f}(ix,:)];positions=[positions;tr(f).S.RAM(ix-1)]; %#ok<AGROW>
 Y=[Y;(tr(f).S.RAM(ix)-tr(f).S.RAM(ix-1))*cfg.Fs]; %#ok<AGROW>
end
mu=mean(X,1);sigma=std(X,0,1);sigma(sigma<1e-10)=1;
m.xMean=zeros(1,9);m.xStd=ones(1,9);m.yMean=0;m.yStd=1;
q=struct('uMean',mu,'uStd',sigma,'xCenter',mean(positions),...
 'xScale',max(std(positions),1e-9),'rateScale',max(std(Y),1e-9),...
 'W',.3*randn(H,4),'b',.1*randn(H,1),'v',zeros(H,1),'bias',0);
Z=[bsxfun(@rdivide,bsxfun(@minus,X,mu),sigma) (positions-q.xCenter)/q.xScale];
A=[tanh(bsxfun(@plus,Z*q.W',q.b')) ones(size(Z,1),1)];
w=(A'*A/size(A,1)+1e-3*eye(H+1))\(A'*(Y/q.rateScale)/size(A,1));q.v=w(1:H);q.bias=w(end);m.dynamicMLP=q;m=bb_ram_bounds(m,tr,cfg);
% Uniform windows spanning every TRAIN flight. All batches advanced together.
U=zeros(T,0,3);target=zeros(T,0);x0=[];per=max(1,floor(cfg.dmlpWindows/numel(tr)));
for f=1:numel(tr)
 first=max(2,cfg.evalStartIndex);last=numel(tr(f).t)-T+1;
 if last<first,error('TRAIN flight too short for dynamic MLP horizon');end
 starts=unique(round(linspace(first,last,min(per,last-first+1))));
 for start=starts
  ix=start:start+T-1;k=size(target,2)+1;
  U(:,k,:)=reshape(bsxfun(@rdivide,bsxfun(@minus,inputs{f}(ix,:),mu),sigma),T,1,3);
  target(:,k)=tr(f).S.RAM(ix);x0(k,1)=tr(f).S.RAM(start-1); %#ok<AGROW>
 end
end
p=[q.W(:);q.b;q.v;q.bias];initial=localLoss(p,m,U,target,x0);
options=optimset('Display','off','MaxFunEvals',cfg.dmlpMaxEvals,'MaxIter',cfg.dmlpMaxEvals,'TolX',1e-4,'TolFun',1e-7);
timer=tic;[best,loss,flag,output]=fminsearch(@(p)localLoss(p,m,U,target,x0),p,options);
if ~isfinite(loss)||loss>initial,best=p;loss=initial;end
m=localUnpack(best,m);seconds=toc(timer);m=bb_ram_bounds(m,tr,cfg);
sse=0;n=0;
for f=1:numel(va)
 p=bb_rollout_rate(m,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=cfg.evalStartIndex:numel(p);
 sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
end
score=sqrt(sse/n);if ~isfinite(score),score=Inf;end
report=table(H,T,size(target,2),sqrt(initial)*q.xScale,sqrt(loss)*q.xScale,score,seconds,flag,output.funcCount,...
 'VariableNames',{'Hidden','Horizon','Windows','InitialWindowRMSE','TrainWindowRMSE','ValidationRMSE','TrainSeconds','ExitFlag','Evaluations'});
end
function m=localUnpack(p,m)
H=numel(m.dynamicMLP.b);q=m.dynamicMLP;q.W=reshape(p(1:4*H),H,4);q.b=p(4*H+(1:H));q.v=p(5*H+(1:H));q.bias=p(end);m.dynamicMLP=q;
end
function loss=localLoss(p,m,U,target,x)
m=localUnpack(p,m);sse=0;
for k=1:size(U,1)
 u=reshape(U(k,:,:),size(U,2),3);x=bb_dmlp_step(m,x,u);
 if isfield(m,'ramBounds'),x=min(max(x,m.ramBounds(1)),m.ramBounds(2));end
 if any(~isfinite(x))||any(abs(x-m.dynamicMLP.xCenter)>20*m.dynamicMLP.xScale),loss=1e100;return;end
 sse=sse+sum(((x-target(k,:)')/m.dynamicMLP.xScale).^2);
end
loss=sse/numel(target);
end
