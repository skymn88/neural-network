function test_pulse_horizon()
c=bb_config();c.makePlots=false;c.evalStartIndex=2;c.pulseProfile=[.1 2;0 5;-.1 4;0 5];
f.name='fixture';f.t=(0:127)'/64;f.S.CoilCurrent=sin(f.t*5);f.S.DDV=f.S.CoilCurrent;
f.S.RAM=3+[0;cumsum(f.S.DDV(2:end))/64];f.S.Deflection=2*f.S.RAM;
m=struct('Fs',64,'kind','FIR','inputNames',{{'DDV'}},'inputLags',0,'outputLags',0,...
 'xMean',0,'xStd',1,'yMean',0,'yStd',1,'weights',[0;1],'beta',0,'rateGain',1,'rateBias',0);
plant.rate=m;plant.ddv=m;plant.ddv.inputNames={'CoilCurrent'};plant.angle=struct('coefficients',[0;2],'scale',1);
fl=bb_pulse_flight(f,c);assert(numel(fl.t)==16*64);
assert(abs(fl.S.CoilCurrent(1)-.1*std(f.S.CoilCurrent))<1e-12);
y=3+[0;cumsum(fl.S.CoilCurrent(2:end))/64];[tab,droop]=bb_pulse_metrics(fl,y,64);
assert(droop<1e-12&&all(abs(tab.DriftPerSecond(strcmp(tab.Phase,'hold')))<1e-12));
configs.Perfect=plant;report=bb_pulse_report(configs,f,c,'');assert(height(report)==8);
r=bb_horizon_report(configs,f,f,c);assert(height(r)==20&&max(r.RMSE)<1e-10);
[p,mask]=bb_reset_rollout(m,f,f.S.DDV,.5,2);assert(max(abs(p(mask)-f.S.RAM(mask)))<1e-10);
c.selectionMode='windowed';c.selectionWindowSeconds=.5;
[score,metrics]=bb_ram_score(m,f,{f.S.DDV},c);assert(score<1e-10&&metrics.FullFreeRunRMSE<1e-10);
c.holdDroopMax=.001;c.selectionPulse=fl;c.selectionPulse.S.DDV=fl.S.CoilCurrent;
bad=m;bad.weights=[1;1];[score,metrics]=bb_ram_score(bad,f,{f.S.DDV},c);
assert(isinf(score)&&~metrics.Eligible&&metrics.HoldDroop>.9);
c.holdDroopMax=Inf;g=f;g.S.RAM=[0;cumsum([ones(63,1);-ones(64,1)])/64];
a=m;a.weights=[0;0];b=m;b.weights=[.5;0];c.selectionWindowSeconds=.25;
[wa,ma]=bb_ram_score(a,g,{g.S.DDV},c);[wb,mb]=bb_ram_score(b,g,{g.S.DDV},c);
assert(mb.FullFreeRunRMSE<ma.FullFreeRunRMSE&&wa<wb,'Full/reset rankings must be distinguishable');
% Cleanup tolerates installations without Simulink; leaves ordinary figures alone.
bb_release_runtime();
end
