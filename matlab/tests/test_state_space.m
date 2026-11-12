function test_state_space()
c=bb_config();c.inputLags=2;c.evalStartIndex=8;c.trainStartIndex=8;c.oeMaxTrainRows=Inf;c.ssMaxEvals=15;
f.name='ss';f.t=(0:79)'/64;f.S.DDV=sin(f.t*7);f.S.RAM=cumsum(f.S.DDV)/64;
m=bb_oe_train(f,c,-.3,2,1e-3);
[n,r,score]=bb_ss_train(m,f,f,{f.S.DDV},c);
assert(isfinite(score)&&r.TrainRMSE<=r.InitialTrainRMSE+1e-12);
g=f;g.S=rmfield(g.S,'RAM');
a=bb_rollout_rate(n,f,0);b=bb_rollout_rate(n,g,0);assert(isequal(a,b));
z=n;z.stateSpace.weights=zeros(1,5);assert(isequal(bb_rollout_rate(z,g,0),bb_rollout_rate(m,g,0)));
x=.4;drive=.7;q=n.stateSpace;q.weights=[.1 -.2 .3 -.1 .2];n.stateSpace=q;
w=tanh(q.weights);v=tanh(drive/q.driveScale);u=(x-q.center)/q.scale;
expected=(1+n.beta/n.Fs)*x+drive/n.Fs+q.scale/n.Fs*(w(1)+w(2)*u+w(3)*tanh(u)+w(4)*v+w(5)*tanh(u)*v);
assert(abs(bb_ram_step(n,x,drive)-expected)<1e-12);
end
