function test_dynamic_mlp()
c=bb_config();c.dmlpHorizon=16;c.dmlpWindows=4;c.dmlpMaxEvals=20;c.evalStartIndex=12;
f.name='dynamic';f.t=(0:119)'/64;f.S.DDV=.2*sin(f.t*4);
f.S.RAM=cumsum(f.S.DDV)/64;f.S.Deflection=2*f.S.RAM;
[m,r,score]=bb_dmlp_train(f,f,{f.S.DDV},c);
assert(isfinite(score)&&r.TrainObjective<=r.InitialObjective+1e-12);
combined=c.dmlpPositionWeight*(r.TrainWindowRMSE/m.dynamicMLP.xScale)^2+...
 c.dmlpMoveRateWeight*(r.MoveRateRMSE/m.dynamicMLP.rateScale)^2+...
 c.dmlpHoldRateWeight*(r.HoldRateRMSE/m.dynamicMLP.rateScale)^2+...
 c.dmlpDisplacementWeight*(r.DisplacementRMSE/m.dynamicMLP.xScale)^2;
assert(abs(combined-r.TrainObjective)<1e-9&&r.HoldSamples+r.MoveSamples==r.Windows*r.Horizon);
g=f;g.S=rmfield(g.S,'RAM');a=bb_rollout_rate(m,f,0);b=bb_rollout_rate(m,g,0);
assert(isequal(a,b),'Inference must not read RAM target');
[~,rate,info]=bb_rollout_rate(m,g,0);assert(isequal(size(rate),size(a))&&size(info.externalInputs,2)==3);
u=bb_predict_batch(m,g);x=0;
for k=2:numel(a),x=bb_dmlp_step(m,x,u(k,:));assert(abs(x-a(k))<1e-12);end
batch=bb_dmlp_step(m,a(1:end-1),u(2:end,:));
assert(max(abs(batch-a(2:end)))<1e-12);
Z=[1 2 3;4 5 6];v=bb_dmlp_inputs(Z);assert(isequal(v,[1 2 -1;4 5 -1]));
% Physical initial state must be retained at sample 1.
offset=bb_rollout_rate(m,g,3);assert(offset(1)==3);
% Full evaluation path including one-step, component and cascade metrics.
f.S.CoilCurrent=f.S.DDV;plant.ddv=bb_fit_candidate(f,c,'NL_OE',{'CoilCurrent'},{'DDV'},'NL');
plant.rate=m;plant.angle=struct('coefficients',[0;2],'scale',1);configs.DMLP=plant;
[d,~]=bb_evaluate_rich(configs,f,f,f,c);assert(height(d)==15&&all(isfinite(d.RMSE)));
[map,~]=bb_rate_diagnostics(plant,f,f,c);assert(width(map)==8);
plant.trainBounds=[-10 10];f.S.RAMSetpoint=f.S.RAM;
ctrl=struct('Fs',64,'lags',2,'taus',1,'kind','FIR','nOutputs',1,...
 'xMean',zeros(1,8),'xStd',ones(1,8),'yMean',.01,'yStd',1,'weights',zeros(9,1));
[p,closedInfo]=bb_closed_loop(ctrl,plant,f,0);g=f;g.S.DDV=closedInfo.ddv;
assert(max(abs(bb_rollout_rate(m,g,0)-p))<1e-10);
end
