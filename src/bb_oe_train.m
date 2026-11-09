function model=bb_oe_train(flights,cfg,beta,lags,ridge)
% FIR + integrator fitted directly to TRAIN position trajectory error.
% No differentiated position target, no output history in network inputs.
if beta>0||1+beta/cfg.Fs<=0||ridge<=0,error('Invalid OE settings');end
if isinf(cfg.oeMaxTrainRows)||cfg.oeZeroRateAtNull||...
        (isfield(cfg,'oeCustom')&&cfg.oeCustom)||isfield(cfg,'oeInputName')
    model=bb_oe_stream(flights,cfg,beta,lags,ridge);return;
end
names={'DDV'};if cfg.usePressure,names=[names {'SupplyPressure','ReturnPressure'}];end
first=cfg.evalStartIndex;count=max(1,floor(cfg.oeMaxTrainRows/numel(flights)));
X=[];yAll=[];
for f=1:numel(flights)
    fl=flights(f);xx=bb_lag_inputs(fl,names,lags);
    ix=localIndices(first,numel(fl.t)-1,count);
    X=[X;xx(ix,:)];yAll=[yAll;fl.S.RAM]; %#ok<AGROW>
end
model.inputBasis='linear';
if isfield(cfg,'oeNonlinear')&&cfg.oeNonlinear,model.inputBasis='nonlinear';end
model.name='FIR_OE';model.kind='FIR';model.mode='mlp';model.Fs=cfg.Fs;
model.inputNames=names;model.outputNames={'RAMRate'};model.nInputs=numel(names);model.nOutputs=1;
model.inputLags=lags;model.outputLags=0;model.xMean=mean(X,1);model.xStd=std(X,0,1);
model.xStd(model.xStd<1e-10)=1;model.yMean=0;model.yStd=std(yAll);
if model.yStd<1e-10,model.yStd=1;end
A=[];Y=[];a=1+beta/cfg.Fs;
for f=1:numel(flights)
    fl=flights(f);xx=bb_lag_inputs(fl,names,lags);
    Z=bsxfun(@rdivide,bsxfun(@minus,xx,model.xMean),model.xStd);
    if strcmp(model.inputBasis,'nonlinear'),Z=[Z tanh(Z) abs(Z)];end
    phi=[ones(size(Z,1),1) Z];phi(1,:)=0;
    response=filter(1/cfg.Fs,[1 -a],phi,[],1);
    k=(0:numel(fl.t)-1)';target=(fl.S.RAM-fl.S.RAM(1)*a.^k)/model.yStd;
    ix=localIndices(first,numel(fl.t)-1,count);
    A=[A;response(ix,:)];Y=[Y;target(ix)]; %#ok<AGROW>
end
scale=sqrt(mean(A.^2,1));scale(scale<1e-12)=1;Q=bsxfun(@rdivide,A,scale);
R=eye(size(Q,2));R(1,1)=0; % Do not regularize the physical rate bias.
w=(Q'*Q/size(Q,1)+ridge*R)\(Q'*Y/size(Q,1));
model.weights=bsxfun(@rdivide,w,scale');model.beta=beta;
model.rateGain=1;model.rateBias=0;model.ridge=ridge;model.trainRows=size(Q,1);
model.objective='position output error';model.stateKind='integrator_or_leaky_integrator';
end
function ix=localIndices(first,last,count)
if last<first,error('Flight too short for common OE evaluation window');end
ix=unique(round(linspace(first,last,min(count,last-first+1))));
end
