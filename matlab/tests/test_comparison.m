function test_comparison()
% Base MATLAB unit checks; no training toolbox required.
r=bb_compare_metrics([0;1],[0;1],1);assert(r(4)==100&&r(5)==0);
r=bb_compare_metrics([0;1],[0;NaN],1);assert(isinf(r(5))&&r(7)==50);
r=bb_compare_metrics([1;1],[2;2],1);assert(isnan(r(4)));
% Analytically known FIR: y(k)=2*u(k)+3*u(k-1)+1.
m.kind='FIR';m.mode='mlp';m.nInputs=1;m.nOutputs=1;
m.inputNames={'CoilCurrent'};m.outputNames={'DDV'};
m.inputLags=1;m.outputLags=0;m.xMean=[0 0];m.xStd=[1 1];
m.yMean=0;m.yStd=1;m.weights=[1;2;3];
f.t=(0:3)';f.S.CoilCurrent=[1;2;4;8];
y=bb_predict_batch(m,f);assert(isequal(y,[6;8;15;29]));
% Target channel deliberately absent: inference must still work.
assert(~isfield(f.S,'DDV'));
m.kind='GP';m.centers=[0 0];m.lengthScale=1;m.alpha=2;
y=bb_forward(m,[0 0;1 0]);assert(max(abs(y-[2;2*exp(-.5)]))<1e-12);
test_split_policy();
fprintf('COMPARISON UNIT CHECKS PASS\n');
end
