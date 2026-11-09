function test_oe_model()
% Exact known integrator, units/offset invariance, no target at inference.
cfg=bb_config();cfg.Fs=64;cfg.usePressure=false;cfg.evalStartIndex=66;cfg.oeMaxTrainRows=10000;
for f=1:3
    t=(0:1999)'/64;u=sin((1:2000)'*(.031+.007*f))+.3*cos((1:2000)'*.113);
    x=zeros(2000,1);x(1)=.01*f;
    for k=2:2000,x(k)=x(k-1)+(.08*u(k)-.04*u(k-1)+.001)/64;end
    fl(f).name=sprintf('T%d',f);fl(f).t=t;fl(f).S.DDV=u;fl(f).S.RAM=x; %#ok<AGROW>
end
m=bb_oe_train(fl(1:2),cfg,0,1,1e-10);
p=bb_rollout_rate(m,fl(3),fl(3).S.RAM(1));assert(max(abs(p-fl(3).S.RAM))<1e-6);
scaled=fl;for f=1:3,scaled(f).S.RAM=1000*fl(f).S.RAM+40;end
ms=bb_oe_train(scaled(1:2),cfg,0,1,1e-10);
ps=bb_rollout_rate(ms,scaled(3),scaled(3).S.RAM(1));assert(max(abs(ps-(1000*p+40)))<1e-6);
noTarget=rmfield(fl(3).S,'RAM');ft=fl(3);ft.S=noTarget;
assert(isequal(bb_rollout_rate(m,ft,fl(3).S.RAM(1)),p));
hold=bb_hold_model(cfg);q=bb_rollout_rate(hold,ft,7);assert(all(q==7));
raw=bb_make_demo();raw=rmfield(raw,'ActuatorForce');prepared=bb_prepare(raw(1),cfg);assert(isfield(prepared.S,'RAM'));
cfg.requireForce=true;threw=false;try,bb_prepare(raw(1),cfg);catch,threw=true;end;assert(threw);
% Known nonlinear input transformation and inference width contract.
nl=m;nl.inputBasis='nonlinear';nl.weights=[.1;.2;.3;.4;.5;.6;.7];
raw=[.2 -.3];Z=(raw-nl.xMean)./nl.xStd;
expected=[1 Z tanh(Z) abs(Z)]*nl.weights*nl.yStd;
assert(abs(bb_forward(nl,raw)-expected)<1e-12);
fprintf('OE / INTEGRATOR / OPTIONAL FORCE CHECKS PASS\n');
end
