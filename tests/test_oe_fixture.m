function test_oe_fixture(path)
% Independent Python-trained OE inference check; not MATLAB training proof.
d=load(path);m=d.model;m.name='FIR_OE';m.kind='FIR';m.mode='mlp';m.Fs=64;
m.inputNames={'DDV'};m.outputNames={'RAMRate'};m.nInputs=1;m.nOutputs=1;m.outputLags=0;m.yMean=0;
f.t=(0:numel(d.u)-1)'/64;f.S.DDV=d.u;
x=bb_rollout_rate(m,f,d.x0);err=max(abs(x-d.expected));assert(err<1e-9);
fprintf('OE FIXTURE PARITY PASS: %.6g\n',err);
end
