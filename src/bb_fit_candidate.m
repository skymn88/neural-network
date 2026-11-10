function model=bb_fit_candidate(flights,cfg,name,inputNames,outputNames,kind)
% Train-only candidate fit. GP uses base MATLAB Cholesky; no new toolbox.
if strcmp(kind,'MLP')
    model=bb_train(flights,cfg,name,inputNames,outputNames,'mlp');
    return;
end
X=[];Y=[];nu=cfg.inputLags;
for f=1:numel(flights)
    fl=flights(f); U=[];T=[];
    for j=1:numel(inputNames), U(:,j)=fl.S.(inputNames{j}); end %#ok<AGROW>
    for j=1:numel(outputNames), T(:,j)=fl.S.(outputNames{j}); end %#ok<AGROW>
    first=nu+1;last=numel(fl.t);
    if isfield(cfg,'trainStartIndex'),first=max(first,cfg.trainStartIndex);end
    if isfield(cfg,'trainEndTrim'),last=last-cfg.trainEndTrim;end
    count=max(1,floor(cfg.maxTrainRows/numel(flights)));
    idx=bb_training_indices(U,first,last,count,cfg,f);
    xx=[];
    for lag=0:nu, xx=[xx U(idx-lag,:)]; end %#ok<AGROW>
    X=[X;xx];Y=[Y;T(idx,:)]; %#ok<AGROW>
end
model.name=name;model.kind=kind;model.mode='mlp';model.Fs=cfg.Fs;
model.inputNames=inputNames;model.outputNames=outputNames;
model.nInputs=numel(inputNames);model.nOutputs=numel(outputNames);
model.inputLags=nu;model.outputLags=0;
model.xMean=mean(X,1);model.xStd=std(X,0,1);
model.yMean=mean(Y,1);model.yStd=std(Y,0,1);
model.xStd(model.xStd<1e-10)=1;model.yStd(model.yStd<1e-10)=1;
X=bsxfun(@rdivide,bsxfun(@minus,X,model.xMean),model.xStd);
Y=bsxfun(@rdivide,bsxfun(@minus,Y,model.yMean),model.yStd);
if strcmp(kind,'NL')
    % Direct DDV output error, no RAM integrator or measured output feedback.
    model.kind='FIR';model.inputBasis='nonlinear';model.tanhScale=1;
    model.objective='direct output error';model.ridge=cfg.ddvRidge;
    X=[X tanh(X) abs(X)];A=[ones(size(X,1),1) X];
    R=eye(size(A,2));R(1,1)=0;
    model.weights=(A'*A/size(A,1)+cfg.ddvRidge*R)\(A'*Y/size(A,1));
elseif strcmp(kind,'FIR')
    A=[ones(size(X,1),1) X]; R=.01*eye(size(A,2));R(1,1)=0;
    model.weights=(A'*A+R)\(A'*Y);
elseif strcmp(kind,'GP')
    points=400;noise=.02;
    if isfield(cfg,'gpMaxPoints'),points=cfg.gpMaxPoints;end
    if isfield(cfg,'gpNoiseVariance'),noise=cfg.gpNoiseVariance;end
    points=min(points,size(X,1));
    if isfield(cfg,'gpCoverage')&&cfg.gpCoverage
        ix=zeros(points,1);dist=inf(size(X,1),1);[~,next]=max(sum(X.^2,2));
        for c=1:points
            ix(c)=next;dist=min(dist,sum(bsxfun(@minus,X,X(next,:)).^2,2));
            dist(ix(1:c))=-1;[~,next]=max(dist);
        end
    else
        ix=unique(round(linspace(1,size(X,1),points)));
    end
    model.centers=X(ix,:);model.lengthScale=sqrt(size(X,2));
    Z=model.centers;
    d=max(0,bsxfun(@plus,sum(Z.^2,2),sum(Z.^2,2)')-2*(Z*Z'));
    K=exp(-.5*d/model.lengthScale^2)+noise*eye(numel(ix));
    C=chol(K,'lower'); model.alpha=C'\(C\Y(ix,:));
else
    error('Unknown candidate: %s',kind);
end
model.trainRows=size(X,1);
end
