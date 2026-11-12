function X=bb_lag_inputs(flight,inputNames,lags,override)
% Current and past EXTERNAL inputs only; no measured target access.
if nargin<4,override=struct();end
N=numel(flight.t);U=zeros(N,numel(inputNames));
for j=1:numel(inputNames)
    name=inputNames{j};
    if isfield(override,name),v=override.(name);else,v=flight.S.(name);end
    if numel(v)~=N,error('Input length mismatch: %s',name);end
    U(:,j)=v(:);
end
if isempty(U)||any(~isfinite(U(:))),error('Empty or nonfinite inference input');end
p=[repmat(U(1,:),lags,1);U];X=zeros(N,(lags+1)*numel(inputNames));
for lag=0:lags
    cols=lag*numel(inputNames)+(1:numel(inputNames));
    X(:,cols)=p(lags+1-lag:end-lag,:);
end
end
