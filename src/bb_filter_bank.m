function [X,state]=bb_filter_bank(U,spec,state)
% Chunk/one-sample equivalent; optional steady IIR start, raw history repeats u(1).
if nargin<3,state=[];end
U=double(U);[N,nu]=size(U);if N==0||any(~isfinite(U(:))),error('Invalid FB input');end
np=numel(spec.poles);nr=size(spec.a,1);L=spec.rawLags;
if isempty(state)
 state.raw=repmat(U(1,:),L,1);state.first=zeros(np,nu);state.second=zeros(2,nu,nr);
 if isfield(spec,'steadyStart')&&spec.steadyStart
  state.first=spec.poles(:)*U(1,:);
  for j=1:nr
   state.second(:,:,j)=[1-spec.b(j);-spec.a(j,3)]*U(1,:);
  end
 end
end
history=[state.raw;U];X=zeros(N,nu*(L+1+np+nr));
for lag=0:L,X(:,lag*nu+(1:nu))=history(L+1-lag:L+N-lag,:);end
state.raw=history(end-L+1:end,:);col=(L+1)*nu;
for j=1:np
 [v,z]=filter(1-spec.poles(j),[1 -spec.poles(j)],U,state.first(j,:),1);
 X(:,col+(1:nu))=v;state.first(j,:)=z;col=col+nu;
end
for j=1:nr
 [v,z]=filter(spec.b(j),spec.a(j,:),U,state.second(:,:,j),1);
 X(:,col+(1:nu))=v;state.second(:,:,j)=z;col=col+nu;
end
end
