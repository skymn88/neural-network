function test_ddv_stream()
c=bb_config();c.inputLags=3;c.fbRawLags=2;c.fbPoles=[.5 .9];c.fbFrequencies=2;c.fbDamping=.7;
c.trainStartIndex=12;c.evalStartIndex=12;c.trainEndTrim=1;c.maxTrainRows=Inf;c.oeChunkRows=17;
f.name='ddv';f.t=(0:179)'/64;f.S.CoilCurrent=sin(f.t*5)+.1*cos(f.t*17);
f.S.DDV=filter(.2,[1 -.8],f.S.CoilCurrent)+.05*tanh(f.S.CoilCurrent);
for kind={'NL','FB'}
 c.ddvAllSamples=false;a=bb_fit_candidate(f,c,'A',{'CoilCurrent'},{'DDV'},kind{1});
 c.ddvAllSamples=true;b=bb_fit_candidate(f,c,'B',{'CoilCurrent'},{'DDV'},kind{1});
 assert(a.trainRows==b.trainRows&&b.trainRows==168);
 assert(max(abs(bb_predict_batch(a,f)-bb_predict_batch(b,f)))<1e-8);
 c.maxTrainRows=2;c.trainingSampling='mixed';c.oeChunkRows=31;
 d=bb_fit_candidate(f,c,'D',{'CoilCurrent'},{'DDV'},kind{1});
 assert(d.trainRows==b.trainRows&&max(abs(bb_predict_batch(d,f)-bb_predict_batch(b,f)))<1e-8);
 c.maxTrainRows=Inf;
end
models.NL_FB=b;models.NL_FB_Sampled=a;
r=bb_ddv_report(models,f,f,f,c);assert(all(isfinite(r.AllSamplesRMSE(1:3))));
% OOF uses the same all-row entry point; perturb held target cannot affect prediction.
f2=f;f2.name='two';f3=f;f3.name='three';tr=[f f2 f3];
[out,~]=bb_ddv_oof(tr,c,'NL_FB');tr(1).S.DDV=tr(1).S.DDV+100;
[changed,~]=bb_ddv_oof(tr,c,'NL_FB');assert(isequal(out(1).S.DDV,changed(1).S.DDV));
% Refit keeps the same all-sample DDV route.
f.S.RAM=cumsum(f.S.DDV)/64;f.S.Deflection=2*f.S.RAM;c.ramTrainInput='measured';
plant.ddv=b;plant.rate=bb_oe_train(f,c,-.3,2,1e-3);
plant.angle=struct('coefficients',[0;2],'degree',1,'scale',1);
[refit,~]=bb_refit_trainval(plant,f,f,c,'NL_FB');
assert(refit.ddv.allSamples&&refit.ddv.trainRows==336);
end
