function model=bb_ctrl_train(tr,cfg,taus,ridge)
% One unified black-box controller, ridge LS; TRAIN feedback noise optional.
if any(taus<=0)||ridge<=0||cfg.ctrlLags<1||cfg.ctrlNoiseStd<0,error('Invalid controller settings');end
model.Fs=cfg.Fs;model.lags=cfg.ctrlLags;model.taus=taus(:)';
old=rng;cleanup=onCleanup(@()rng(old));rng(cfg.seed); %#ok<NASGU>
X=[];Y=[];
for f=1:numel(tr)
 fl=tr(f);if ~isfield(fl.S,'RAMSetpoint'),error('RAMSetpoint required for controller');end
 noisy=fl.S.RAM+cfg.ctrlNoiseStd*randn(size(fl.S.RAM));state=[];
 count=max(1,floor(cfg.maxTrainRows/numel(tr)));last=numel(fl.t)-1;
 ix=unique(round(linspace(cfg.evalStartIndex,last,min(count,last-cfg.evalStartIndex+1))));
 selected=false(numel(fl.t),1);selected(ix)=true;
 for k=1:numel(fl.t)
  prev=max(1,k-1);d=0;i=0;if k>1,d=fl.S.DDV(prev);i=fl.S.CoilCurrent(prev);end
  e=fl.S.RAMSetpoint(k)-noisy(prev);
  [z,state]=bb_ctrl_features(e,d,i,state,model);
  if selected(k),X=[X;z];Y=[Y;fl.S.CoilCurrent(k)];end %#ok<AGROW>
 end
end
model.kind='FIR';model.nOutputs=1;model.xMean=mean(X,1);model.xStd=std(X,0,1);model.xStd(model.xStd<1e-10)=1;
model.yMean=mean(Y);model.yStd=std(Y);if model.yStd<1e-10,model.yStd=1;end
Z=bsxfun(@rdivide,bsxfun(@minus,X,model.xMean),model.xStd);A=[ones(size(X,1),1) Z];
R=eye(size(A,2));R(1,1)=0;
model.weights=(A'*A/size(A,1)+ridge*R)\(A'*((Y-model.yMean)/model.yStd)/size(A,1));
model.ridge=ridge;model.noiseStd=cfg.ctrlNoiseStd;model.trainRows=size(X,1);
end
