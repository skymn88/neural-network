function test_filter_bank()
c=bb_config();c.fbRawLags=2;c.fbPoles=[0 .5 .9];c.fbFrequencies=2;c.fbDamping=.3;
c.inputLags=2;c.evalStartIndex=12;c.trainStartIndex=12;c.oeChunkRows=17;c.oeFilterBank=true;
spec=bb_fb_spec(c);u=[sin((1:160)'*.12) cos((1:160)'*.07)];X=bb_filter_bank(u,spec);state=[];parts=[];
for k=1:13:160,[v,state]=bb_filter_bank(u(k:min(k+12,160),:),spec,state);parts=[parts;v];end %#ok<AGROW>
assert(max(abs(parts(:)-X(:)))<1e-12);assert(all(abs(roots(spec.a(1,:)))<1));
state=[];parts=[];for k=1:160,[v,state]=bb_filter_bank(u(k,:),spec,state);parts=[parts;v];end %#ok<AGROW>
assert(max(abs(parts(:)-X(:)))<1e-12);
f.name='bank';f.t=(0:159)'/64;f.S.CoilCurrent=u(:,1);f.S.DDV=u(:,2);
f.S.RAM=cumsum(u(:,2))/64;f.S.Deflection=2*f.S.RAM;f.S.RAMSetpoint=f.S.RAM+.1;
m=bb_oe_train(f,c,-.3,2,1e-3);c.oeChunkRows=1000;n=bb_oe_train(f,c,-.3,2,1e-3);
a=bb_rollout_rate(m,f,0);b=bb_rollout_rate(n,f,0);assert(max(abs(a-b))<1e-8);
g=f;g.S=rmfield(g.S,'RAM');assert(isequal(a,bb_rollout_rate(m,g,0)));
[best,r,score]=bb_fb_search(f,f,{f.S.DDV},c,-.3,1e-3);assert(height(r)==1&&isfinite(score));
plant.rate=best;plant.ddv=bb_fit_candidate(f,c,'NL_FB',{'CoilCurrent'},{'DDV'},'FB');plant.trainBounds=[-100 100];
ctrl=struct('Fs',64,'lags',2,'taus',1,'kind','FIR','nOutputs',1,...
 'xMean',zeros(1,8),'xStd',ones(1,8),'yMean',.01,'yStd',1,'weights',zeros(9,1));[p,info]=bb_closed_loop(ctrl,plant,f,0);
assert(all(isfinite(p)));
 g=f;g.S.CoilCurrent=info.current;assert(max(abs(bb_predict_batch(plant.ddv,g)-info.ddv))<1e-10);
 g.S.DDV=info.ddv;assert(max(abs(bb_rollout_rate(plant.rate,g,0)-p))<1e-9);
models.NL_FB=plant.ddv;report=bb_ddv_report(models,f,f,f,c);assert(height(report)==3&&all(isfinite(report.RMSE)));
% Opt-in steady start: constant input has no IIR startup transient.
c.fbSteadyStart=true;c.fbUniquePoles=true;spec=bb_fb_spec(c);
assert(~any(spec.poles==0));U=ones(25,1)*.7;X=bb_filter_bank(U,spec);
assert(max(abs(X(:)-.7))<1e-12);
state=[];parts=[];for k=1:25,[z,state]=bb_filter_bank(U(k),spec,state);parts=[parts;z];end %#ok<AGROW>
assert(max(abs(parts(:)-X(:)))<1e-12);
c.fbTanhSearch=true;c.nlTanhScales=[.5 1];
[~,scaleReport,~]=bb_fb_search(f,f,{f.S.DDV},c,-.3,1e-3);
assert(height(scaleReport)==2&&isequal(sort(scaleReport.TanhScale),[.5;1]));

end
