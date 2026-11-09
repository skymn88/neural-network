function model=bb_train(flights,cfg,name,inputNames,outputNames,mode)
% Train a self-contained black-box component using TRAIN flights only.
% mode: 'mlp' (input history) or 'narx' (input + measured output history).
if isempty(flights), error('No training flights'); end
if ~any(strcmp(mode,{'mlp','narx'})), error('Unknown network mode %s',mode); end
nu=cfg.inputLags; ny=0;
if strcmp(mode,'narx'), ny=cfg.outputLags; end
nU=numel(inputNames); nY=numel(outputNames);
D=(nu+1)*nU+ny*nY;
maxPer=max(1,floor(cfg.maxTrainRows/numel(flights)));
X=zeros(0,D); Y=zeros(0,nY);
for f=1:numel(flights)
    U=localCols(flights(f),inputNames);
    T=localCols(flights(f),outputNames);
    first=max(nu,ny)+1;last=size(U,1);
    if isfield(cfg,'trainStartIndex'),first=max(first,cfg.trainStartIndex);end
    if isfield(cfg,'trainEndTrim'),last=last-cfg.trainEndTrim;end
    idx=bb_training_indices(U,first,last,maxPer,cfg,f);
    xx=zeros(numel(idx),D); yy=zeros(numel(idx),nY);
    for k=1:numel(idx)
        t=idx(k);
        uLag=U(t:-1:t-nu,:);
        feat=reshape(uLag.',1,[]);
        if ny>0
            yLag=T(t-1:-1:t-ny,:);
            feat=[feat reshape(yLag.',1,[])]; %#ok<AGROW>
        end
        xx(k,:)=feat; yy(k,:)=T(t,:);
    end
    X=[X;xx]; Y=[Y;yy]; %#ok<AGROW>
end
if size(X,1)<30, error('%s: too few training samples',name); end
xMean=mean(X,1); xStd=std(X,0,1);
yMean=mean(Y,1); yStd=std(Y,0,1);
xStd(xStd<1e-10)=1; yStd(yStd<1e-10)=1;
Xn=bsxfun(@rdivide,bsxfun(@minus,X,xMean),xStd);
Yn=bsxfun(@rdivide,bsxfun(@minus,Y,yMean),yStd);
rng(cfg.seed);
trainFcn='trainscg';if isfield(cfg,'trainFcn'),trainFcn=cfg.trainFcn;end
net=fitnet(cfg.hiddenSizes,trainFcn);
net.inputs{1}.processFcns={};
net.outputs{net.numLayers}.processFcns={};
net.divideFcn='dividetrain';
net.trainParam.epochs=cfg.epochs;
net.trainParam.showWindow=false;
net.trainParam.showCommandLine=false;
net=configure(net,Xn.',Yn.');
net=train(net,Xn.',Yn.');
model.name=name; model.mode=mode; model.Fs=cfg.Fs;
model.inputNames=inputNames; model.outputNames=outputNames;
model.nInputs=nU; model.nOutputs=nY;
model.inputLags=nu; model.outputLags=ny;
model.xMean=xMean;model.xStd=xStd;model.yMean=yMean;model.yStd=yStd;
model.layers=struct('W',{},'b',{},'activation',{});
for k=1:net.numLayers
    if k==1, W=net.IW{1,1}; else, W=net.LW{k,k-1}; end
    model.layers(k).W=W;
    model.layers(k).b=net.b{k};
    model.layers(k).activation=net.layers{k}.transferFcn;
end
idx=1:min(24,size(X,1));
yPortable=bb_forward(model,X(idx,:));
yToolbox=bsxfun(@plus,bsxfun(@times,net(Xn(idx,:).').',yStd),yMean);
err=max(abs(yPortable(:)-yToolbox(:)));
if ~isfinite(err) || err>1e-7*max(1,max(abs(yToolbox(:))))
    error('%s: MATLAB/portable model mismatch: %.3g',name,err);
end
model.trainRows=size(X,1);model.parityError=err;
fprintf('  %-14s %-5s | windows=%d | portable parity max=%.3g\n', ...
    name,upper(mode),model.trainRows,err);
end

function X=localCols(f,names)
X=zeros(numel(f.t),numel(names));
for j=1:numel(names)
    if ~isfield(f.S,names{j}), error('Missing %s in %s',names{j},f.name); end
    X(:,j)=f.S.(names{j});
end
end
