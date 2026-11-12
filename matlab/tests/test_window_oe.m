function test_window_oe()
c=bb_config();c.evalStartIndex=8;c.oeChunkRows=17;c.oeWindowSeconds=.5;c.oeTransitionWeight=2;
c.fbRawLags=2;c.fbPoles=[.5 .9];c.fbFrequencies=2;c.fbDamping=.7;
f.name='window';f.t=(0:149)'/64;f.S.CoilCurrent=sin(f.t*7);f.S.DDV=f.S.CoilCurrent;
f.S.RAM=2+cumsum(f.S.DDV)/64+.1*sin(f.t*2);
for family=1:3
 q=c;
 if family==1,q.oeNonlinear=true;
 elseif family==2,q.oeFilterBank=true;
 else,q.oeCustom=true;q.oeZeroRateAtNull=true;q.featureSpec=struct('basis','hold','parameter',.03,'force',false,'pressure',false);end
 m=bb_oe_train(f,q,-.3,2,1e-3);N=numel(f.t);W=32;a=1-.3/64;
 X=bb_oe_features(m,f,(1:N)');Z=bsxfun(@rdivide,bsxfun(@minus,X,m.xMean),m.xStd);
 if strcmp(m.inputBasis,'nonlinear'),Z=[Z tanh(Z/m.tanhScale) abs(Z)];end
 if ~m.noIntercept,Z=[ones(N,1) Z];end
 A=zeros(size(Z));Y=zeros(N,1);mask=false(N,1);
 for b=1:W:N
  ix=b:min(N,b+W-1);block=Z(ix,:);block(1,:)=0;
  A(ix,:)=filter(1/64,[1 -a],block);Y(ix)=(f.S.RAM(ix)-f.S.RAM(b)*a.^((0:numel(ix)-1)'))/m.yStd;
  mask(ix(2:end))=true;
 end
 mask(1:c.evalStartIndex-1)=false;mask(end)=false;
 moving=bb_transition_mask(f.S.RAM,64,m.transitionThreshold,c.oeTransitionSeconds);
 w=1+2*double(moving(mask));A=A(mask,:);Y=Y(mask);total=sum(w);
 G=A'*bsxfun(@times,A,w);h=A'*(Y.*w);scale=sqrt(diag(G)'/total);scale(scale<1e-12)=1;
 Q=bsxfun(@rdivide,bsxfun(@rdivide,G,scale),scale')/total;R=eye(size(Q));if ~m.noIntercept,R(1,1)=0;end
 dense=((Q+1e-3*R)\(h./scale'/total))./scale';
 assert(max(abs(dense-m.weights))<1e-8,'Weighted window OE dense parity');
 q.oeChunkRows=41;n=bb_oe_train(f,q,-.3,2,1e-3);
 assert(max(abs(bb_rollout_rate(n,f,2)-bb_rollout_rate(m,f,2)))<1e-8);
end
% Explicit design-state reset is independent of tool chunk boundaries.
Z=reshape(sin(1:180),60,3);[all,~]=bb_oe_window_filter(Z,(1:60)',.99,64,16,[]);
state=[];parts=[];
for b=1:7:60,ix=(b:min(60,b+6))';[v,state]=bb_oe_window_filter(Z(ix,:),ix,.99,64,16,state);parts=[parts;v];end %#ok<AGROW>
assert(max(abs(parts(:)-all(:)))<1e-12);
% Each family's search must propagate training windows and the selected metric.
c.oeWindowSeconds=[.25 .5];c.oeWindowSearch=true;c.oeWindowBetas=[0 -.03];
c.selectionMode='windowed';c.selectionWindowSeconds=.5;c.nlRefine=false;
c.holdLags=0;c.holdDeadbandFractions=0;c.holdRidge=1e-3;
[~,r,score]=bb_fb_search(f,f,{f.S.DDV},c,[0 -.03],1e-3);
assert(all(ismember(r.WindowSeconds,[.25 .5]))&&abs(score-min(r.ValidationRMSE))<1e-12);
[~,r,score]=bb_nl_search(f,f,{f.S.DDV},c,2,[0 -.03],1e-3);
assert(all(isfinite(r.ResetWindowRMSE))&&abs(score-min(r.ValidationRMSE))<1e-12);
[~,r,score]=bb_hold_search(f,f,{f.S.DDV},c);
assert(height(r)==4&&abs(score-min(r.ValidationRMSE))<1e-12);

end
