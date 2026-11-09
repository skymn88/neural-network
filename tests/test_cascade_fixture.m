function test_cascade_fixture(path)
% Python-trained exported weights; this verifies inference, not MATLAB training.
d=load(path);c=d.selected;f.t=(0:numel(d.current)-1)'/64;f.S.CoilCurrent=d.current;
u=bb_predict_batch(c.ddv,f);
x=bb_rollout_rate(c.rate,f,d.x0,struct('DDV',u));
a=polyval(flipud(c.angle.coefficients),x/c.angle.scale);
e=[max(abs(u-d.expectedDDV)) max(abs(x-d.expectedRAM)) max(abs(a-d.expectedAngle))];
assert(all(e<1e-9),'Cascade fixture mismatch');
fprintf('CASCADE FIXTURE PARITY PASS: %.6g %.6g %.6g\n',e);
end
