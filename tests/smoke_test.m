function out=smoke_test(runUnits)
% Reduced training + complete reporting path. No fidelity acceptance claim.
if nargin<1,runUnits=true;end
if runUnits
    test_comparison();test_rate_model();test_rich_model();test_oe_model();test_compact_summary();
end
out=run_rich_comparison('synthetic',true);
assert(exist(fullfile(out,'comparison.xlsx'),'file')==2);
assert(exist(fullfile(out,'trained_models.mat'),'file')==2);
assert(exist(fullfile(out,'summary_short.txt'),'file')==2);
t=readtable(fullfile(out,'comparison.xlsx'),'Sheet','Summary');
assert(all(ismember({'NL_OE','HoldInitial'},unique(t.Model))));
assert(all(t.CoveragePct==100),'Nonfinite smoke predictions');
fprintf('SMOKE PASS (code flow only): %s\n',out);
end
