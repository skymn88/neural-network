function test_static_null()
c=bb_config();c.evalStartIndex=10;c.nullCenter=false;c.oeWindowSearch=false;c.selectionMode='fullFreeRun';c.makePlots=false;
f.name='null';f.t=(0:399)'/64;f.S.DDV=.2+.3*sin(f.t*3);
f.S.CoilCurrent=f.S.DDV-.2;f.S.RAM=5+[0;cumsum(2*(f.S.DDV(2:end)-.2))/64];f.S.Deflection=2*f.S.RAM;
[center,info]=bb_train_null(f,c);assert(info.RootFound&&abs(center-.2)<.03);
[root,found,count]=bb_grid_null([-1;0;1],[-2;0;2],.1);assert(found&&root==0&&count==1);
[root,found,~]=bb_grid_null([-1;0;1],[1;2;3],0);assert(~found&&isnan(root));
m=struct('Fs',64,'kind','FIR','inputNames',{{'DDV'}},'inputLags',0,'outputLags',0,...
 'xMean',0,'xStd',1,'yMean',0,'yStd',1,'weights',[0;2],'beta',-.3,'rateGain',1,'rateBias',0,'ddvCenter',.2);
[g,net]=bb_static_drive(m,.2,5);assert(abs(g)<1e-12&&abs(net+1.5)<1e-12);
plant.rate=m;plant.ddv=m;plant.ddv=rmfield(plant.ddv,{'ddvCenter','beta','rateGain','rateBias'});
plant.ddv.inputNames={'CoilCurrent'};plant.ddv.weights=[.2;1];plant.angle=struct('coefficients',[0;2],'scale',1);
[r,curve]=bb_static_null(plant,f,c);assert(abs(r.DriveNullDDV-.2)<1e-12&&abs(r.DDVAtZeroCurrent-.2)<1e-12);
assert(height(curve)==c.nullGridPoints); % net null may lie outside TRAIN range
plant.trainBounds=[-10 10];f.S.RAMSetpoint=f.S.RAM;
ctrl=struct('Fs',64,'lags',2,'taus',1,'kind','FIR','nOutputs',1,...
 'xMean',zeros(1,8),'xStd',ones(1,8),'yMean',0,'yStd',1,'weights',zeros(9,1));
[closed,trace]=bb_closed_loop(ctrl,plant,f,5);
assert(max(abs(bb_rollout_rate(m,f,5,struct('DDV',trace.ddv))-closed))<1e-10);
report=bb_pulse_report(struct('A',plant),f,c,'');assert(all(report.Beta==-.3)&&all(report.OutputIntercept==0));
% Same offset in TRAIN and inference, including steady filter-bank features.
for bank=[false true]
 q=c;q.nullCenter=true;q.nullDDVCenter=.2;q.oeFilterBank=bank;q.fbRawLags=2;q.oeNonlinear=true;
 a=bb_oe_train(f,q,0,2,1e-3);shift=f;shift.S.DDV=shift.S.DDV-.2;q.nullCenter=false;
 b=bb_oe_train(shift,q,0,2,1e-3);
 assert(max(abs(bb_rollout_rate(a,f,5)-bb_rollout_rate(b,shift,5)))<1e-9);
 [av,~]=bb_static_drive(a,.35,5);[bv,~]=bb_static_drive(b,.15,5);assert(abs(av-bv)<1e-9);
end
% Dynamic training also centers exactly once and remains target-independent.
c.nullCenter=true;c.nullDDVCenter=.2;c.dmlpHorizon=16;c.dmlpWindows=4;c.dmlpMaxEvals=15;
[n,report,~]=bb_dmlp_train(f,f,{f.S.DDV},c);assert(n.ddvCenter==.2&&report.TrainObjective<=report.InitialObjective+1e-12);
g=f;g.S=rmfield(g.S,'RAM');assert(isequal(bb_rollout_rate(n,f,5),bb_rollout_rate(n,g,5)));
end
