function test_compact_summary()
Model={'FIR_OE';'FIR_OE';'FIR_OE'};Stage=repmat({'RAM_Cascade'},3,1);Split={'train';'validation';'test'};
FitPct=[90;80;-Inf];RMSE=[1;2;Inf];OneStepFitPct=nan(3,1);TrainStdNRMSEpct=[10;20;Inf];CoveragePct=[100;100;50];
s=table(Model,Stage,Split,FitPct,RMSE,OneStepFitPct,TrainStdNRMSEpct,CoveragePct);
selection=struct('DDV','MLP','RAM','FIR_OE');
text=evalc('compact=bb_print_summary(s,selection);');
assert(~isempty(strfind(text,'-Inf'))&&~isempty(strfind(text,'Inf'))); %#ok<STREMP>
assert(compact.Fit_Train(1)==90&&compact.Fit_Val(1)==80&&isinf(compact.Fit_Test(1)));
lines=regexp(text,'\n','split');assert(max(cellfun(@numel,lines))<=100);
fprintf('COMPACT REPORT CHECKS PASS\n');
end
