function test_cascade_fixture(path)
% Python-trained exported weights; this verifies inference, not MATLAB training.
d=load(path);c=d.selected;f.t=(0:numel(d.current)-1)'/64;f.S.CoilCurrent=d.current;
u=bb_predict_batch(c.ddv,f);
x=bb_rollout_rate(c.rate,f,d.x0,struct('DDV',u));
a=polyval(flipud(c.angle.coefficients),x/c.angle.scale);
actual={u,x,a};expected={d.expectedDDV,d.expectedRAM,d.expectedAngle};e=zeros(1,3);tol=e;
for j=1:3
 assert(numel(actual{j})==numel(expected{j}),'Fixture length mismatch');
 e(j)=max(abs(actual{j}(:)-expected{j}(:)));
 tol(j)=1e-9+1e-8*max(abs(expected{j}(:)));
end
assert(all(isfinite(e))&&all(e<=tol),...
 'Cascade fixture mismatch: DDV/RAM/Angle errors [%g %g %g], tolerances [%g %g %g]. Check which bb_forward -all.',e(1),e(2),e(3),tol(1),tol(2),tol(3));
fprintf('CASCADE FIXTURE PARITY PASS: %.6g %.6g %.6g\n',e);
end
