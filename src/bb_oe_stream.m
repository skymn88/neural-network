function model=bb_oe_stream(flights,cfg,beta,lags,ridge)
% Two-pass, blockwise OE normal equations; O(features^2 + chunk*features) memory.
if beta>0||1+beta/cfg.Fs<=0||ridge<=0,error('Invalid OE settings');end
if cfg.oeChunkRows<1||cfg.oeChunkRows~=floor(cfg.oeChunkRows),error('Invalid chunk size');end
model.name='FIR_OE';model.kind='FIR';model.mode='mlp';model.Fs=cfg.Fs;
model.inputNames={'DDV'};
if cfg.usePressure,model.inputNames=[model.inputNames {'SupplyPressure','ReturnPressure'}];end
model.outputNames={'RAMRate'};model.nInputs=numel(model.inputNames);model.nOutputs=1;
model.inputLags=lags;model.outputLags=0;model.inputBasis='linear';
if isfield(cfg,'oeNonlinear')&&cfg.oeNonlinear,model.inputBasis='nonlinear';end
model.noIntercept=cfg.oeZeroRateAtNull;
first=cfg.evalStartIndex;count=max(1,floor(cfg.oeMaxTrainRows/numel(flights)));
nx=0;mx=[];sx=[];ny=0;my=0;sy=0;indices=cell(size(flights));
for f=1:numel(flights)
 fl=flights(f);last=numel(fl.t)-1;
 if last<first,error('Flight too short for common OE evaluation window');end
 if isinf(count),ix=first:last;else,ix=unique(round(linspace(first,last,min(count,last-first+1))));end
 indices{f}=ix;
 for b=1:cfg.oeChunkRows:numel(ix)
  ii=ix(b:min(end,b+cfg.oeChunkRows-1));X=bb_oe_features(model,fl,ii);
  [nx,mx,sx]=localMoments(nx,mx,sx,X);
 end
 for b=1:cfg.oeChunkRows:numel(fl.t)
  ii=b:min(numel(fl.t),b+cfg.oeChunkRows-1);[ny,my,sy]=localMoments(ny,my,sy,fl.S.RAM(ii));
 end
end
model.xMean=mx;model.xStd=sqrt(sx/max(1,nx-1));model.xStd(model.xStd<1e-10)=1;
% Null DDV must stay zero: no centering or intercept for the constrained variant.
if model.noIntercept,model.xMean(:)=0;end
model.yMean=0;model.yStd=sqrt(sy/max(1,ny-1));if model.yStd<1e-10,model.yStd=1;end
G=[];h=[];n=0;a=1+beta/cfg.Fs;
for f=1:numel(flights)
 fl=flights(f);state=[];ix=indices{f};
 for b=1:cfg.oeChunkRows:numel(fl.t)
  ii=(b:min(numel(fl.t),b+cfg.oeChunkRows-1))';
  X=bb_oe_features(model,fl,ii);Z=bsxfun(@rdivide,bsxfun(@minus,X,model.xMean),model.xStd);
  if strcmp(model.inputBasis,'nonlinear'),Z=[Z tanh(Z) abs(Z)];end
  if ~model.noIntercept,Z=[ones(size(Z,1),1) Z];end
  if b==1,Z(1,:)=0;state=zeros(1,size(Z,2));end
  [A,state]=filter(1/cfg.Fs,[1 -a],Z,state,1);
  keep=ismember(ii,ix);A=A(keep,:);k=ii(keep)-1;
  Y=(fl.S.RAM(ii(keep))-fl.S.RAM(1)*a.^k)/model.yStd;
  if isempty(G),G=zeros(size(Z,2));h=zeros(size(Z,2),1);end
  G=G+A'*A;h=h+A'*Y;n=n+sum(keep);
 end
end
scale=sqrt(diag(G)'/n);scale(scale<1e-12)=1;
Q=bsxfun(@rdivide,bsxfun(@rdivide,G,scale),scale')/n;
R=eye(size(Q));if ~model.noIntercept,R(1,1)=0;end
w=(Q+ridge*R)\(h./scale'/n);model.weights=w./scale';
model.beta=beta;model.rateGain=1;model.rateBias=0;model.ridge=ridge;model.trainRows=n;
model.objective='position output error';model.stateKind='integrator_or_leaky_integrator';
end
function [n,m,s]=localMoments(n,m,s,X)
k=size(X,1);if k==0,return;end
v=mean(X,1);ss=sum(bsxfun(@minus,X,v).^2,1);
if n==0,n=k;m=v;s=ss;return;end
d=v-m;s=s+ss+d.^2*(n*k/(n+k));m=m+d*(k/(n+k));n=n+k;
end
