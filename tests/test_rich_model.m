function test_rich_model()
% No training/toolbox needed: analytical calibration and mixed-window sampling.
m.kind='FIR';m.nInputs=1;m.nOutputs=1;m.inputNames={'DDV'};
m.outputNames={'RAMRate'};m.inputLags=0;m.outputLags=0;
m.xMean=0;m.xStd=1;m.yMean=0;m.yStd=1;m.weights=[0;1];m.beta=-.025;m.Fs=64;
f.t=(0:999)'/64;f.S.DDV=.01*sin((0:999)'*.07)+.02*cos((0:999)'*.021);
teacher=m;teacher.rateGain=1.3;teacher.rateBias=.0002;
f.S.RAM=bb_rollout_rate(teacher,f,.03);
[g,b]=bb_rate_calibrate(m,f);assert(abs(g-1.3)<1e-9&&abs(b-.0002)<1e-11);
m.rateGain=g;m.rateBias=b;pred=bb_rollout_rate(m,f,.03);
assert(max(abs(pred-f.S.RAM))<1e-10);
f.S.RAM(:)=NaN;pred2=bb_rollout_rate(m,f,.03);assert(isequal(pred,pred2));
c=bb_config();c.trainingSampling='mixed';U=zeros(5000,1);U(1001:1020)=1;U(3001:3010)=-1;
state=rng;ix=bb_training_indices(U,66,4999,1000,c,1);assert(isequal(state,rng));
assert(numel(ix)==1000&&numel(unique(ix))==1000&&min(ix)>=66&&max(ix)<=4999);
assert(isequal(ix,bb_training_indices(U,66,4999,1000,c,1)));
assert(sum(ix>=981&ix<=1041)>20);
fprintf('RICH UNIT CHECKS PASS\n');
end
