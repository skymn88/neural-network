function test_transition_weighting()
c=bb_config();assert(strcmp(c.ramTrainInput,'oof')&&isinf(c.oeMaxTrainRows)&&~c.holdBaselines);
c.inputLags=1;c.evalStartIndex=6;c.oeMaxTrainRows=10000;c.nlRefine=false;
f.name='weighted';f.t=(0:119)'/64;u=sin((1:120)'*.11);f.S.DDV=u;f.S.CoilCurrent=u;
f.S.RAM=[zeros(30,1);linspace(0,1,10)';ones(80,1)];f.S.Deflection=2*f.S.RAM;
m0=bb_oe_train(f,c,-.3,1,1e-3);s0=bb_oe_stream(f,c,-.3,1,1e-3);
assert(max(abs(bb_rollout_rate(m0,f,0)-bb_rollout_rate(s0,f,0)))<1e-9);
c.oeTransitionWeight=2;m=bb_oe_stream(f,c,-.3,1,1e-3);
X=bb_lag_inputs(f,{'DDV'},1);Z=(X-m.xMean)./m.xStd;phi=[ones(120,1) Z];phi(1,:)=0;
a=1+m.beta/64;A=filter(1/64,[1 -a],phi,[],1);Y=(f.S.RAM-f.S.RAM(1)*a.^(0:119)')/m.yStd;
mask=bb_transition_mask(f.S.RAM,64,m.transitionThreshold,c.oeTransitionSeconds);ix=c.evalStartIndex:119;
w=1+2*double(mask(ix));A=A(ix,:);Y=Y(ix);W=sum(w);
scale=sqrt(sum(bsxfun(@times,A.^2,w),1)/W);scale(scale<1e-12)=1;Q=A./scale;
R=eye(size(Q,2));R(1,1)=0;expected=((Q'*bsxfun(@times,Q,w)/W+1e-3*R)\(Q'*(w.*Y)/W))./scale';
assert(max(abs(expected-m.weights))<1e-8);assert(m.weightSum==W);
c.oeTransitionWeight=[0 2 5];[best,r,score,all]=bb_nl_search(f,f,{u},c,1,-.3,1e-3);
assert(isequal(unique(r.TransitionWeight),[0;2;5])&&numel(fieldnames(all))==3);
assert(score<=r.ValidationRMSE(r.TransitionWeight==0));assert(ismember(best.transitionWeight,[0 2 5]));
end
