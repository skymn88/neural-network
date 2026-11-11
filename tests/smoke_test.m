function out=smoke_test(runUnits)
% Reduced training + complete reporting path. No fidelity acceptance claim.
if nargin<1,runUnits=true;end
if runUnits
    test_comparison();test_rate_model();test_rich_model();test_oe_model();test_compact_summary();
end
out=run_rich_comparison('synthetic',true,'plant',struct('oeMaxTrainRows',700,'ramTrainInput','measured','makePlots',false,'dmlpWindows',16,'dmlpHorizon',32,'dmlpMaxEvals',20));
assert(exist(fullfile(out,'comparison.xlsx'),'file')==2);
assert(exist(fullfile(out,'trained_models.mat'),'file')==2);
assert(exist(fullfile(out,'summary_short.txt'),'file')==2);
saved=load(fullfile(out,'selection.mat'),'summary');t=saved.summary;
[~,sheets]=xlsfinfo(fullfile(out,'comparison.xlsx'));
layoutCfg=bb_config();if ~layoutCfg.legacySheets,assert(numel(sheets)<=6);end
assert(any(ismember(t.Model,{'NL_OE','NL_FB'})));
c=bb_config();
if ~c.holdBaselines,assert(~any(ismember(t.Model,{'HoldInitial','TrainMean'})));end
assert(all(t.CoveragePct==100),'Nonfinite smoke predictions');
fprintf('SMOKE PASS (code flow only): %s\n',out);
end
