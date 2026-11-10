function test_workbook()
c=bb_config();c.legacySheets=false;s=struct('RAM','A','DDV','NL_OE','RAMBeta',-.3,'RAMLags',8);
Model={'A';'A';'A'};Stage=repmat({'RAM_Cascade'},3,1);Split={'train';'validation';'test'};
FitPct=[80.1234;70;60];RMSE=[1;.00000123456;.5];
r.Summary=table(Model,Stage,Split,FitPct,RMSE);r.PerFlight=r.Summary;r.Compact=bb_wide_summary(r.Summary,s,c);
assert(r.Compact.Fit_Train(1)==80.12&&r.Compact.RMSE_Val(1)>0);
r.RunInfo=table({'test'},'VariableNames',{'Source'});r.Split=table({'F'},{'train'},'VariableNames',{'Flight','Split'});
r.NL_Selection=table(-.3,1,'VariableNames',{'Beta','ValidationRMSE'});r.BandGain=table(.4,'VariableNames',{'StdRatio'});
t=struct('DDVTraining',1,'OOF',2,'RAMSearch',3,'RAM_NL_FB',2,'ReportWrite',0,'Total',0);
file=[tempname '.xlsx'];cleanup=onCleanup(@()localDelete(file)); %#ok<NASGU>
layout=bb_write_workbook(file,r,s,c,t);t.ReportWrite=4;t.Total=10;bb_stamp_timings(file,layout,t,c);
[~,sheets]=xlsfinfo(file);assert(numel(sheets)==5&&all(ismember({'Summary','PerFlight','Diagnostics','Search','Config'},sheets)));
a=readcell(file,'Sheet','Summary');assert(strcmp(a{layout.summaryTimeRow,1},'Stage'));
assert(strcmp(a{layout.summaryTimeRow+1,1},'DDVTraining')&&a{layout.summaryTimeRow+1,2}==1);
localDelete(file);c.legacySheets=true;layout=bb_write_workbook(file,r,s,c,t);bb_stamp_timings(file,layout,t,c);
a=readtable(file,'Sheet','Summary');assert(isequal(a.FitPct,FitPct));
% Guard the previously introduced concatenated end/if token explicitly.
text=fileread(which('bb_print_summary'));assert(isempty(regexp(text,'\<endif\>','once')));
end
function localDelete(file)
if exist(file,'file'),delete(file);end
end
