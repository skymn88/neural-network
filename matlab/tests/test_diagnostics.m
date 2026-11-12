function test_diagnostics()
cfg=bb_config();cfg.diagnostics=true;cfg.alignmentMaxLag=12;
rng(7);u=randn(300,1);y=[zeros(5,1);u(1:end-5)];
f.name='delay';f.t=(0:299)'/64;f.S.CoilCurrent=u;f.S.DDV=y;f.S.RAM=cumsum(y)/64;
r=bb_alignment_report(f,f,f,cfg);assert(all(r.LagSamples(strcmp(r.Pair,'Current_DDV'))==5));
f.S.CoilCurrent=y;f.S.DDV=u;r=bb_alignment_report(f,f,f,cfg);assert(r.LagSamples(1)==-5);
f.S.CoilCurrent=zeros(300,1);r=bb_alignment_report(f,f,f,cfg);assert(isnan(r.LagSamples(1)));
[s,w,n]=bb_hold_drift(3*(0:99)'/64,true(100,1),64);assert(abs(s-3)<1e-12&&abs(w-3)<1e-12&&n==1);
[s,~,n]=bb_hold_drift(zeros(10,1),false(10,1),64);assert(isnan(s)&&n==0);
Model={'A';'B';'HoldInitial'};Stage=repmat({'RAM_Cascade'},3,1);Split=repmat({'test'},3,1);
FitPct=[1;2;3];RMSE=[1;2;3];OneStepFitPct=FitPct;TrainStdNRMSEpct=FitPct;CoveragePct=FitPct;
t=table(Model,Stage,Split,FitPct,RMSE,OneStepFitPct,TrainStdNRMSEpct,CoveragePct);
cfg.printTestForAll=false;c=bb_print_summary(t,struct('RAM','A'),'',cfg);
assert(c.Fit_Test(1)==1&&isnan(c.Fit_Test(2))&&c.Fit_Test(3)==3);
end
