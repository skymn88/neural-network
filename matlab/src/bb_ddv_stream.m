function m=bb_ddv_stream(flights,cfg,name,inputNames,outputNames,kind)
% Uniform ALL eligible TRAIN rows, two passes, bounded feature memory.
if cfg.oeChunkRows<1||cfg.oeChunkRows~=floor(cfg.oeChunkRows),error('Invalid DDV chunk size');end
m=struct('name',name,'kind','FIR','mode','mlp','Fs',cfg.Fs,...
 'inputNames',{inputNames},'outputNames',{outputNames},'inputLags',cfg.inputLags,...
 'outputLags',0,'nInputs',numel(inputNames),'nOutputs',numel(outputNames),...
 'inputBasis','nonlinear','tanhScale',1,'objective','direct output error','ridge',cfg.ddvRidge);
if strcmp(kind,'FB'),m.filterBank=bb_fb_spec(cfg);m.inputLags=cfg.fbRawLags;end
count=0;mu=[];ss=[];
for pass=1:2
 G=[];h=[];
 for f=1:numel(flights)
  fl=flights(f);first=m.inputLags+1;last=numel(fl.t);state=[];
  if isfield(cfg,'trainStartIndex'),first=max(first,cfg.trainStartIndex);end
  if isfield(cfg,'trainEndTrim'),last=last-cfg.trainEndTrim;end
  for b=1:cfg.oeChunkRows:numel(fl.t)
   ii=(b:min(numel(fl.t),b+cfg.oeChunkRows-1))';
   if isfield(m,'filterBank')
    U=zeros(numel(ii),m.nInputs);
    for j=1:m.nInputs,U(:,j)=fl.S.(inputNames{j})(ii);end
    [X,state]=bb_filter_bank(U,m.filterBank,state);
   else,X=bb_oe_features(m,fl,ii);end
   keep=ii>=first&ii<=last;X=X(keep,:);ii=ii(keep);if isempty(ii),continue;end
   Y=zeros(numel(ii),m.nOutputs);
   for j=1:m.nOutputs,Y(:,j)=fl.S.(outputNames{j})(ii);end
   if pass==1
    V=[X Y];n=size(V,1);v=mean(V,1);s=sum(bsxfun(@minus,V,v).^2,1);
    if count==0,mu=v;ss=s;count=n;
    else,d=v-mu;ss=ss+s+d.^2*(count*n/(count+n));mu=mu+d*n/(count+n);count=count+n;end
   else
    Z=bsxfun(@rdivide,bsxfun(@minus,X,m.xMean),m.xStd);
    A=[ones(size(Z,1),1) Z tanh(Z) abs(Z)];
    Y=bsxfun(@rdivide,bsxfun(@minus,Y,m.yMean),m.yStd);
    if isempty(G),G=zeros(size(A,2));h=zeros(size(A,2),m.nOutputs);end
    G=G+A'*A;h=h+A'*Y;
   end
  end
 end
 if pass==1
  if count<2,error('Insufficient DDV training rows');end
  sigma=sqrt(ss/(count-1));sigma(sigma<1e-10)=1;k=numel(mu)-m.nOutputs;
  m.xMean=mu(1:k);m.xStd=sigma(1:k);m.yMean=mu(k+1:end);m.yStd=sigma(k+1:end);
 end
end
R=eye(size(G));R(1,1)=0;m.weights=(G/count+cfg.ddvRidge*R)\(h/count);
m.trainRows=count;m.allSamples=true;
end
