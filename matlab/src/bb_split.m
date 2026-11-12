function [tr,va,te,manifest,description] = bb_split(flights,cfg)
% Deterministic flight split policy:
% 1-3 flights: each flight is time-split 70/20/10 with guard gaps.
% 4-9 flights: (N-2)/1/1 whole-flight split.
% >=10 flights: approximately 70/20/10 whole-flight split.
% All downstream models receive the same tr/va/te arrays.
n=numel(flights);
if n<1, error('bb_split: no flights supplied.'); end
rows=cell(0,3);
if n<=3
    tr=flights; va=flights; te=flights;
    for f=1:n
        [a,b,c] = localTimeSplit(flights(f),cfg);
        tr(f)=a; va(f)=b; te(f)=c;
        rows(end+1,:)={flights(f).name,'Train',numel(a.t)}; %#ok<AGROW>
        rows(end+1,:)={flights(f).name,'Validation',numel(b.t)}; %#ok<AGROW>
        rows(end+1,:)={flights(f).name,'Test',numel(c.t)}; %#ok<AGROW>
    end
    description=sprintf('%d flight(s): EACH flight time-split 70/20/10 with guards; NOT independent-flight validation.',n);
else
    if n<10
        nTrain=n-2; nVal=1; nTest=1;
    elseif n==11
        nTrain=7;nVal=2;nTest=2;
    else
        nVal=max(1,round(0.20*n));
        nTest=max(1,round(0.10*n));
        nTrain=n-nVal-nTest;
    end
    if nTrain<1, error('bb_split: not enough training flights.'); end
    oldState=rng;
    rng(cfg.seed);
    ix=randperm(n);
    rng(oldState);
    tr=flights(ix(1:nTrain));
    va=flights(ix(nTrain+(1:nVal)));
    te=flights(ix(nTrain+nVal+(1:nTest)));
    for f=1:numel(tr), rows(end+1,:)={tr(f).name,'Train',numel(tr(f).t)}; end %#ok<AGROW>
    for f=1:numel(va), rows(end+1,:)={va(f).name,'Validation',numel(va(f).t)}; end %#ok<AGROW>
    for f=1:numel(te), rows(end+1,:)={te(f).name,'Test',numel(te(f).t)}; end %#ok<AGROW>
    description=sprintf('%d independent flights: %d train / %d validation / %d test (seed=%d).', ...
        n,nTrain,nVal,nTest,cfg.seed);
end
manifest=cell2table(rows,'VariableNames',{'Flight','Split','Samples'});
end

function [a,b,c] = localTimeSplit(f,cfg)
N=numel(f.t);
guard=max(round(cfg.guardSeconds*cfg.Fs),max(cfg.inputLags,cfg.outputLags)+2);
k1=floor(0.70*N); k2=floor(0.90*N);
iTr=1:(k1-guard);
iVa=(k1+guard+1):(k2-guard);
iTe=(k2+guard+1):N;
need=max(cfg.inputLags,cfg.outputLags)+8;
if min([numel(iTr),numel(iVa),numel(iTe)])<need
    error('bb_split: %s too short for 70/20/10 with guard %.3g s.',f.name,cfg.guardSeconds);
end
a=localSlice(f,iTr,'train');
b=localSlice(f,iVa,'validation');
c=localSlice(f,iTe,'test');
end

function a=localSlice(a,idx,suffix)
a.t=a.t(idx)-a.t(idx(1));
names=fieldnames(a.S);
for i=1:numel(names)
    x=a.S.(names{i});
    a.S.(names{i})=x(idx,:);
end
a.name=[a.name '_' suffix];
end
