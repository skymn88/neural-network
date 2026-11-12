function test_wh_model()
c=bb_config();c.inputLags=1;c.evalStartIndex=6;c.whMaxEvals=30;c.whMaxIter=10;
map=struct('knots',[-3;-1;0;1;3],'values',[-2;0;0;0;4]);
z=bb_wh_map(map,[-99;-.5;0;.5;99]);assert(isequal(z,[-2;0;0;0;4]));
assert(all(diff(bb_wh_map(map,linspace(-4,4,100)'))>=0));
m=bb_hold_model(c);m.beta=-.3;r=sin((1:200)'*.13);p=bb_wh_rollout(m,r,2);q=zeros(200,1);q(1)=2;
for k=2:200,q(k)=(1+m.beta/64)*q(k-1)+r(k)/64;end
assert(max(abs(p-q))<1e-12&&p(1)==2);
f.name='wh';f.t=(0:199)'/64;f.S.CoilCurrent=r;f.S.DDV=r;
f.S.RAM=[zeros(30,1);linspace(0,1,10)';ones(160,1)];f.S.Deflection=2*f.S.RAM;
base=bb_oe_train(f,c,-.3,1,1e-3);[m,report]=bb_wh_train(base,f,c);
assert(all(diff(m.wh.knots)>0)&&all(diff(m.wh.values)>=0));assert(bb_wh_map(m.wh,0)==0);
assert(m.beta<=0&&m.beta>-64&&report.TrainRMSE<=report.InitialTrainRMSE+1e-10);
g=f;g.S=rmfield(g.S,'RAM');assert(isequal(bb_rollout_rate(m,f,0),bb_rollout_rate(m,g,0)));
[fixed,~]=bb_wh_train(base,f,c,true);assert(fixed.beta==base.beta);
plant.rate=m;plant.ddv=bb_fit_candidate(f,c,'NL_OE',{'CoilCurrent'},{'DDV'},'NL');
[rateMap,steps]=bb_rate_diagnostics(plant,f,f,c);
assert(height(rateMap)==2*c.rateMapBins&&sum(rateMap.Count)==2*(200-c.evalStartIndex+1));
assert(height(steps)>=1&&height(steps)<=5&&steps.MeasuredDuration10to90(1)>0);
end
