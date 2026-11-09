function test_oe_stream()
c=bb_config();c.evalStartIndex=8;c.oeMaxTrainRows=1e6;
for j=1:3
 f(j).name=sprintf('F%d',j);f(j).t=(0:199)'/64; %#ok<AGROW>
 u=sin((1:200)'*.13*j);f(j).S.DDV=u;f(j).S.CoilCurrent=u;
 f(j).S.RAM=filter(1/64,[1 -.99],u);
end
m=bb_oe_train(f,c,-.64,3,1e-3);c.oeMaxTrainRows=Inf;c.oeChunkRows=17;
n=bb_oe_train(f,c,-.64,3,1e-3);
assert(n.trainRows==3*(199-8+1));
assert(max(abs(bb_rollout_rate(m,f(1),0)-bb_rollout_rate(n,f(1),0)))<1e-9);
x=bb_oe_features(n,f(1),[1 2 70]);all=bb_lag_inputs(f(1),{'DDV'},3);assert(isequal(x,all([1 2 70],:)));
c.oeZeroRateAtNull=true;n=bb_oe_train(f,c,0,3,1e-3);assert(bb_forward(n,zeros(1,4))==0);
[o,r]=bb_ddv_oof(f,c,'FIR_rich');assert(height(r)==3);
g=f;g(1).S.DDV=g(1).S.DDV+100;[og,~]=bb_ddv_oof(g,c,'FIR_rich');
assert(isequal(o(1).S.DDV,og(1).S.DDV)); % held target never enters its predictor
[o,r]=bb_ddv_oof(f(1:2),c,'FIR_rich');assert(isempty(r)&&isequal(o,f(1:2)));
end
