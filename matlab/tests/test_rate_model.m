function test_rate_model()
% Base MATLAB tests: known discrete state solution and input-only inference.
m.kind='FIR';m.nInputs=1;m.nOutputs=1;m.inputNames={'DDV'};
m.outputNames={'RAMRate'};m.inputLags=0;m.outputLags=0;
m.xMean=0;m.xStd=1;m.yMean=0;m.yStd=1;m.weights=[0;2];
m.beta=-.5;m.Fs=64;
f.t=(0:127)'/64;f.S.DDV=zeros(128,1);
x=bb_rollout_rate(m,f,.01);
expected=.01*(1-.5/64).^(0:127)';assert(max(abs(x-expected))<1e-12);
% Zero input response needs initial state, not measured future RAM.
assert(~isfield(f.S,'RAM'));
f.S.DDV(:)=.1;x=bb_rollout_rate(m,f,0);
expected=.4*(1-(1-.5/64).^(0:127)');assert(max(abs(x-expected))<1e-12);
f.S.RAM=1e9*ones(128,1);x2=bb_rollout_rate(m,f,0);
assert(isequal(x,x2),'Inference read measured RAM unexpectedly');
x3=bb_rollout_rate(m,f,.005);
assert(max(abs(x3-x-.005*(1-.5/64).^(0:127)'))<1e-12);
m.beta=1;threw=false;try,bb_rollout_rate(m,f,0);catch,threw=true;end
assert(threw,'Unstable beta accepted');
fprintf('RATE MODEL UNIT CHECKS PASS\n');
end
