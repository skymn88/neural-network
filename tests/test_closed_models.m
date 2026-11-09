function test_closed_models()
c=bb_config();c.evalStartIndex=20;c.inputLags=2;c.ctrlLags=2;c.ctrlTaus=1;c.ctrlRidge=1e-3;c.oeMaxTrainRows=Inf;
f.name='closed';f.t=(0:119)'/64;u=.1*sin((1:120)'*.12);
f.S.CoilCurrent=u;f.S.DDV=u;f.S.RAM=cumsum(u)/64;f.S.RAMSetpoint=f.S.RAM+.02;
plant.ddv=bb_fit_candidate(f,c,'FIR',{'CoilCurrent'},{'DDV'},'FIR');plant.rate=bb_oe_train(f,c,-.1,2,1e-3);
plant.trainBounds=[-1 1];ctrl=bb_ctrl_train(f,c,1,1e-3);
[z,state]=bb_ctrl_features(2,3,4,[],ctrl);assert(z(1)==2&&state.ddv(1)==3&&state.current(1)==4);
[p,a]=bb_closed_loop(ctrl,plant,f,0);g=f;g.S=rmfield(g.S,{'RAM','DDV','CoilCurrent'});
[q,b]=bb_closed_loop(ctrl,plant,g,0);assert(isequaln(p,q)&&isequaln(a.current,b.current));
plant.trainBounds=[-.00001 .00001];[~,a]=bb_closed_loop(ctrl,plant,g,1);assert(a.diverged);
[c1,selection]=bb_closed_train(f,f,plant,c,2,-.1,1e-3);assert(height(selection)==2);
a=bb_rollout_rate(c1.setpoint,f,0);b=bb_rollout_rate(c1.setpoint,g,0);assert(isequaln(a,b));
[s,r]=bb_closed_report(c1,f,f,f,c);assert(height(s)==6&&height(r)==6);
c.ctrlNoiseStd=.01;a=bb_ctrl_train(f,c,1,1e-3);b=bb_ctrl_train(f,c,1,1e-3);assert(isequal(a.weights,b.weights));
raw=bb_make_demo();c.useSetpoint=true;failed=false;try,bb_prepare(raw(1),c);catch,failed=true;end;assert(failed);
end
