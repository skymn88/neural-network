function out=bb_run_checks()
% One-click check entry. Restores working folder and path on failure or success.
paths=bb_paths();folder=paths.root;previousDir=pwd;previousPath=path;
cleanup=onCleanup(@()localRestore(previousDir,previousPath)); %#ok<NASGU>
cd(folder);addpath(folder,'-begin');
if ~exist(paths.results,'dir'),mkdir(paths.results);end
root=fullfile(paths.results,['checks_' datestr(now,'yyyymmdd_HHMMSS')]);out=root;k=1;
while exist(out,'dir'),out=sprintf('%s_%02d',root,k);k=k+1;end
mkdir(out);diary(fullfile(out,'checks.log'));diary on;
fprintf('PROJECT CHECKS | MATLAB %s | %s\n',version,datestr(now));
try
    test_dynamic_mlp();test_ddv_stream();test_hold_model();test_state_space();test_raw_fixture();test_diagnostics();test_oe_stream();test_structural_models();test_ram_loo();test_closed_models();test_nl_focus();test_nl_extensions();test_refit_leak();test_transition_weighting();test_wh_model();test_filter_bank();test_band_gain();test_workbook();
    test_comparison(); % includes split-policy checks N=1..20
    test_rate_model();test_rich_model();test_oe_model();test_compact_summary();
    test_cascade_fixture(fullfile(paths.fixtures,'cascade.mat'));
    test_oe_fixture(fullfile(paths.fixtures,'oe.mat'));
    checkCfg=bb_config();
    if ~checkCfg.nlFocus&&exist('fitnet','file')~=2,error('Legacy smoke needs Deep Learning Toolbox');end
    smokeResults=smoke_test(false);
    save(fullfile(out,'checks_pass.mat'),'smokeResults');
    fprintf('ALL CHECKS AND SMOKE PASS\nSmoke results: %s\nLog: %s\n',smokeResults,out);
catch problem
    fprintf(2,'CHECK FAILED: %s\n',problem.message);
    fprintf(2,'%s\n',getReport(problem,'extended','hyperlinks','off'));
    rethrow(problem);
end
end
function localRestore(folder,oldPath)
diary off;cd(folder);path(oldPath);
end
