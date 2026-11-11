function outDir=run_rich_comparison(source,quick,mode,overrides)
% No NARX. Select on validation, then evaluate test. Bundled synthetic test is a reused regression case.
if nargin<1,source='synthetic';end
if nargin<2,quick=false;end
if nargin<3,mode='plant';end
if nargin<4,overrides=struct();end
if ~ismember(mode,{'plant','closed'}),error('Mode must be plant or closed');end
runTimer=tic;timings=struct('Preprocessing',0,'DDVTraining',0,'OOF',0,'RAMSearch',0,'RAM_NL_OE',0,'RAM_NL_FB',0,'WH',0,'Refit',0,'ClosedTraining',0,'Evaluation',0,'Diagnostics',0,'Plots',0,'ReportWrite',0,'Total',0);
reports=struct();stageTimer=tic;
cfg=bb_config();names=fieldnames(overrides);
for j=1:numel(names)
    if ~isfield(cfg,names{j}),error('Unknown cfg field: %s',names{j});end
    cfg.(names{j})=overrides.(names{j});
end
if cfg.nlStateSpace&&(cfg.nlRefitTrainVal||cfg.nlStateLeak||cfg.reportLoo),error('Disable refit/stateLeak/reportLoo for the isolated state-space comparison');end
if cfg.nlWiener&&cfg.clampRAM,error('NL_WH vectorized experiment requires clampRAM=false');end
if ~cfg.nlFocus&&(cfg.nlRefitTrainVal||cfg.nlStateLeak||cfg.nlJointSearch||cfg.nlWiener)
    error('Refit, state leakage and joint search require nlFocus=true');
end
if strcmp(mode,'closed'),cfg.useSetpoint=true;end
cfg.quick=quick;cfg.trainStartIndex=2*cfg.inputLags+2;cfg.trainEndTrim=1;
paths=bb_paths();
if strcmp(source,'synthetic'),d=load(paths.synthetic);Raw=d.Raw;
elseif strcmp(source,'user'),Raw=load_user_flights();
else,error('Source must be synthetic or user');end
flights=bb_prepare(Raw(1),cfg);for i=2:numel(Raw),flights(i)=bb_prepare(Raw(i),cfg);end
if strcmp(source,'synthetic')
    tr=flights(d.trainIDs);va=flights(d.validationIDs);te=flights(d.testIDs);
    ids={d.trainIDs,d.validationIDs,d.testIDs};splits={'train','validation','test'};rr={};
    for s=1:3,for i=ids{s}(:)',rr(end+1,:)={flights(i).name,splits{s},numel(flights(i).t)};end;end %#ok<AGROW>
    manifest=cell2table(rr,'VariableNames',{'Flight','Split','Samples'});
else
    [tr,va,te,manifest,description]=bb_split(flights,cfg);fprintf('%s\n',description);
end
% Common evaluation window, including the longest candidate's input memory.
allSets=[tr va te];minLength=min(arrayfun(@(f)numel(f.t),allSets));
oeLags=cfg.oeLags;
if quick,oeLags=[16 32];end
oeLags=oeLags(oeLags+cfg.inputLags+2 < minLength-8);
if isempty(oeLags),error('Splits too short for requested candidate memories');end
cfg.evalStartIndex=max(2*cfg.inputLags+2,max(oeLags)+cfg.inputLags+2);
cfg.trainStartIndex=cfg.evalStartIndex;
root=fullfile(cfg.outputRoot,[cfg.runPrefix '_' datestr(now,'yyyymmdd_HHMMSS')]);outDir=root;k=1;
while exist(outDir,'dir'),outDir=sprintf('%s_%02d',root,k);k=k+1;end
mkdir(outDir);timings.Preprocessing=toc(stageTimer);stageTimer=tic;
labels={'MLP_basic','MLP_more_training','MLP_quasiNewton','FIR_rich','GP_coverage'};
if cfg.nlFocus,labels={'NL_OE'};end
if cfg.ddvFilterBank,labels=[labels {'NL_FB'}];end
ddvModels=struct();rates=struct();choices={};best=Inf;start=cfg.evalStartIndex;
for j=1:numel(labels)
    methodTimer=tic;label=labels{j};[c,kind]=bb_rich_config(cfg,label);
    ddvModels.(label)=bb_fit_candidate(tr,c,label,{'CoilCurrent'},{'DDV'},kind);
    sse=0;n=0;
    for f=1:numel(va)
        p=bb_predict_batch(ddvModels.(label),va(f));idx=start:numel(p);
        sse=sse+sum((p(idx)-va(f).S.DDV(idx)).^2);n=n+numel(idx);
    end
    timings.(['DDV_' label])=toc(methodTimer);
    score=sqrt(sse/n);choices(end+1,:)={'DDV',label,NaN,false,1,0,score}; %#ok<AGROW>
    fprintf('DDV %s validation RMSE %.6g\n',label,score);
    if score<best,best=score;chosenDDV=label;end
end
if ~isfinite(best),error('All DDV validation candidates failed');end
vDDV=cell(size(va));for f=1:numel(va),vDDV{f}=bb_predict_batch(ddvModels.(chosenDDV),va(f));end
timings.DDVTraining=toc(stageTimer);stageTimer=tic;
ramTr=tr;oofReport=table();
if ~ismember(cfg.ramTrainInput,{'measured','oof'}),error('Invalid ramTrainInput');end
if strcmp(cfg.ramTrainInput,'oof')||cfg.reportLoo
    [oofTr,oofReport]=bb_ddv_oof(tr,cfg,chosenDDV);
    if strcmp(cfg.ramTrainInput,'oof'),ramTr=oofTr;end
end
timings.OOF=toc(stageTimer);stageTimer=tic;
% Retain DDV search, prune RAM candidates after upstream selection.
labels={'FIR_rich'};
if cfg.nlFocus,labels={};end
bestAll=Inf;
for j=1:numel(labels)
    label=labels{j};[c,kind]=bb_rich_config(cfg,label);betas=cfg.betaCandidates;best=Inf;
    if quick,betas=-.025;end
    for beta=betas
        model=bb_rate_train(tr,c,kind,beta);[gain,bias]=bb_rate_calibrate(model,tr,start);
        variants=[1 0;gain bias];
        for v=1:size(variants,1)
            model.rateGain=variants(v,1);model.rateBias=variants(v,2);model=bb_ram_bounds(model,tr,cfg);sse=0;n=0;
            for f=1:numel(va)
                p=bb_rollout_rate(model,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));idx=start:numel(p);
                sse=sse+sum((p(idx)-va(f).S.RAM(idx)).^2);n=n+numel(idx);
            end
            score=sqrt(sse/n);choices(end+1,:)={'RAM',label,beta,v==2,model.rateGain,model.rateBias,score}; %#ok<AGROW>
            fprintf('RAM %s beta %.5g calibration %d val RMSE %.6g\n',label,beta,v==2,score);
            if score<best,best=score;rates.(label)=model;end
            if score<bestAll,bestAll=score;chosenRate=label;end
        end
    end
end
% Output-error FIR alternative: fit complete position trajectories directly.
oeChoices={};bestOE=Inf;oeBetas=cfg.betaCandidates;oeRidge=cfg.oeRidge;
if quick
    oeBetas=[0 -.025];oeRidge=1e-3;
    if ~isinf(cfg.oeMaxTrainRows),cfg.oeMaxTrainRows=700;end
end
if ~cfg.nlFocus||cfg.nlFIRReference
for beta=oeBetas
 for lag=oeLags
  for ridge=oeRidge
    baseCfg=cfg;baseCfg.oeTransitionWeight=0;baseCfg.oeZeroRateAtNull=false;
    model=bb_oe_train(ramTr,baseCfg,beta,lag,ridge);model=bb_ram_bounds(model,tr,cfg);sse=0;n=0;worst=0;
    for f=1:numel(va)
        p=bb_rollout_rate(model,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));idx=start:numel(p);
        e=p(idx)-va(f).S.RAM(idx);sse=sse+sum(e.^2);n=n+numel(e);
        worst=max(worst,sqrt(mean(e.^2)));
    end
    score=sqrt(sse/n);oeChoices(end+1,:)={beta,lag,ridge,score,worst}; %#ok<AGROW>
    if isfinite(score)&&score<bestOE,bestOE=score;rates.FIR_OE=model;end
    if ~cfg.nlFocus&&isfinite(score)&&score<bestAll,bestAll=score;chosenRate='FIR_OE';end
  end
 end
 fprintf('FIR_OE beta %.5g complete; best validation RMSE %.6g\n',beta,bestOE);
end
if ~isfield(rates,'FIR_OE'),error('No finite OE validation model');end
labels=[labels {'FIR_OE'}];
end
% Nonlinear finite-memory input map + position output-error fit (not NARX).
nlReport=table();weightedRates=struct();
if ~cfg.nlFocus||~cfg.nlFilterBank||cfg.nlReference||cfg.nlWiener
methodTimer=tic;
[rates.NL_OE,nlReport,nlBest,weightedRates]=bb_nl_search(ramTr,va,vDDV,cfg,oeLags,oeBetas,oeRidge);
if nlBest<bestAll,bestAll=nlBest;chosenRate='NL_OE';end
labels=[labels {'NL_OE'}];
timings.RAM_NL_OE=toc(methodTimer);
end
if cfg.oeZeroRateAtNull&&~cfg.nlFocus
    zeroBest=Inf;
    for beta=oeBetas
     for lag=oeLags
      for ridge=oeRidge
       zeroCfg=cfg;zeroCfg.oeTransitionWeight=0;model=bb_oe_train(ramTr,zeroCfg,beta,lag,ridge);model=bb_ram_bounds(model,tr,cfg);sse=0;n=0;
       for f=1:numel(va)
        p=bb_rollout_rate(model,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=start:numel(p);
        sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
       end
       score=sqrt(sse/n);
       if score<zeroBest,zeroBest=score;rates.FIR_OE_Zero=model;end
       if score<bestAll,bestAll=score;chosenRate='FIR_OE_Zero';end
      end
     end
    end
    if ~isfield(rates,'FIR_OE_Zero'),error('No finite null-rate candidate');end
    labels=[labels {'FIR_OE_Zero'}];
end


extraLabels={};extraReport=table();
if cfg.hamEnabled&&~cfg.nlFocus,extraLabels={'HAM_OE'};end
baseLabels=[{'FIR_OE','NL_OE'} extraLabels];
if cfg.nlFocus,baseLabels={'NL_OE'};end
if cfg.useForce,extraLabels=[extraLabels strcat(baseLabels,'_F')];end
if cfg.usePressure&&cfg.pressureInteractions,extraLabels=[extraLabels strcat(baseLabels,'_P')];end
if cfg.useForce&&cfg.usePressure&&cfg.pressureInteractions,extraLabels=[extraLabels strcat(baseLabels,'_F_P')];end
for j=1:numel(extraLabels)
 label=extraLabels{j};
 [model,report,score]=bb_oe_search(ramTr,va,vDDV,cfg,label,oeLags,oeBetas,oeRidge);
 rates.(label)=model;extraReport=[extraReport;report]; %#ok<AGROW>
 if score<bestAll,bestAll=score;chosenRate=label;end
end
labels=[labels extraLabels];
if ~isempty(extraReport),reports.StructuralSelection=extraReport;end
if cfg.nlFocus&&cfg.nlFilterBank
    methodTimer=tic;
    [fb,fbReport,fbScore]=bb_fb_search(ramTr,va,vDDV,cfg,oeBetas,oeRidge);
    timings.RAM_NL_FB=toc(methodTimer);
    rates.NL_FB=fb;labels=[labels {'NL_FB'}];reports.FB_Selection=fbReport;
    if fbScore<bestAll,bestAll=fbScore;chosenRate='NL_FB';end
end
if cfg.nlFocus&&cfg.nlHold
    methodTimer=tic;
    [holdModel,holdReport,holdScore]=bb_hold_search(ramTr,va,vDDV,cfg);
    timings.RAM_NL_HOLD=toc(methodTimer);rates.NL_HOLD=holdModel;
    labels=[labels {'NL_HOLD'}];reports.Hold_Selection=holdReport;
    if holdScore<bestAll,bestAll=holdScore;chosenRate='NL_HOLD';end
end
if cfg.nlFocus&&cfg.nlStateSpace
    methodTimer=tic;
    [ss,ssReport,ssScore]=bb_ss_train(rates.(chosenRate),ramTr,va,vDDV,cfg);
    timings.RAM_NL_SS=toc(methodTimer);rates.NL_SS=ss;labels=[labels {'NL_SS'}];
    reports.SS_Selection=ssReport;
    if ssScore<bestAll,bestAll=ssScore;chosenRate='NL_SS';end
end
if cfg.nlFocus&&cfg.nlStateLeak
    [leak,leakReport,leakScore]=bb_state_leak_fit(rates.(chosenRate),ramTr,va,vDDV,cfg);
    reports.StateLeak=leakReport;
    if isfield(leak,'stateLeak')&&leakScore<bestAll
        rates.NL_OE_State=leak;labels=[labels {'NL_OE_State'}];
        bestAll=leakScore;chosenRate='NL_OE_State';
    end
end
timings.RAMSearch=toc(stageTimer);stageTimer=tic;
if cfg.nlFocus&&cfg.nlWiener
    whBase=rates.NL_OE;
    [wh,whReport]=bb_wh_train(whBase,ramTr,cfg);sse=0;n=0;
    for f=1:numel(va)
        p=bb_rollout_rate(wh,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=start:numel(p);
        sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
    end
    whScore=sqrt(sse/n);whReport.ValidationRMSE=whScore;
    whReport.Selected=isfinite(whScore)&&whScore<bestAll;
    rates.NL_WH=wh;labels=[labels {'NL_WH'}];
    if whReport.Selected,bestAll=whScore;chosenRate='NL_WH';end
    reports.WH_Selection=whReport;
    reports.WH_Map=table(wh.wh.knots,wh.wh.values,'VariableNames',{'RawRate','MappedRate'});
end
timings.WH=toc(stageTimer);stageTimer=tic;
seedRows={cfg.seed,bestAll};

if ~isfinite(bestAll),error('All validation models failed');end
angle=bb_angle_select(tr,va,start);
selection=struct('DDV',chosenDDV,'RAM',chosenRate,'criterion','validation cascade RAM RMSE','seed',cfg.seed,'smokeMode',logical(quick),'source',source);
selectedRate=rates.(chosenRate);selection.RAMBeta=selectedRate.beta;
selection.RAMLags=selectedRate.inputLags;
if isfield(selectedRate,'ridge'),selection.RAMRidge=selectedRate.ridge;end
if isfield(selectedRate,'transitionWeight'),selection.TransitionWeight=selectedRate.transitionWeight;end
if isfield(selectedRate,'tanhScale'),selection.RAMTanhScale=selectedRate.tanhScale;end
% No test outputs were used above.
configs=struct();
for j=1:numel(labels)
    label=labels{j};configs.(label)=struct('ddv',ddvModels.(chosenDDV),'rate',rates.(label),'angle',angle);
end
if cfg.holdBaselines
    configs.HoldInitial=struct('ddv',ddvModels.(chosenDDV),'rate',bb_hold_model(cfg),'angle',angle);
    configs.TrainMean=struct('ddv',ddvModels.(chosenDDV),'rate',bb_train_mean(tr,cfg),'angle',angle);
end
timings.Evaluation=timings.Evaluation+toc(stageTimer);stageTimer=tic;
if cfg.nlDiagnostics
    diagnosticReport=nlReport;
    if strcmp(chosenRate,'NL_FB')||isempty(diagnosticReport)
        diagnosticReport=reports.FB_Selection;
        diagnosticReport.Search=repmat({'coarse'},height(diagnosticReport),1);
        diagnosticReport.Lags=repmat(cfg.fbRawLags,height(diagnosticReport),1);
        diagnosticReport.TanhScale=ones(height(diagnosticReport),1);
    end
    if strcmp(chosenRate,'NL_HOLD')
        diagnosticReport=reports.Hold_Selection;
        diagnosticReport.Search=repmat({'coarse'},height(diagnosticReport),1);
        diagnosticReport.TanhScale=ones(height(diagnosticReport),1);
    end
    [curve,holds,equilibrium]=bb_nl_diagnostics(diagnosticReport,configs.(chosenRate),tr,va,te,cfg);
    reports.BetaCurve=curve;
    reports.HoldVariability=holds;
    reports.NullEquilibrium=equilibrium;
    reports.BandGain=bb_band_gain(configs.(chosenRate),va,te,cfg);
    [rateMap,steps]=bb_rate_diagnostics(configs.(chosenRate),tr,va,cfg);
    reports.RateMap=rateMap;
    reports.StepSmoothing=steps;
end
timings.Diagnostics=timings.Diagnostics+toc(stageTimer);
refitModels=struct();
if cfg.nlFocus&&cfg.nlRefitTrainVal
    refitModels.Original_TRAIN=configs.(chosenRate);
    stageTimer=tic;
    [refitModels.Refit_TRAIN_VAL,refitInfo]=bb_refit_trainval(configs.(chosenRate),tr,va,cfg,chosenDDV);
    timings.Refit=toc(stageTimer);stageTimer=tic;
    ec=cfg;ec.evalTestOnly=true;
    [refitDetails,refitSummary]=bb_evaluate_rich(refitModels,tr,va,te,ec);
    reports.RefitTest=refitSummary;
    reports.RefitPerFlight=refitDetails;
    timings.Evaluation=timings.Evaluation+toc(stageTimer);
    reports.RefitInfo=refitInfo;
end


stageTimer=tic;
[details,summary]=bb_evaluate_rich(configs,tr,va,te,cfg);
timings.Evaluation=timings.Evaluation+toc(stageTimer);
if strcmp(mode,'closed')
    stageTimer=tic;
    [closed,closedSelection]=bb_closed_train(tr,va,configs.(chosenRate),cfg,oeLags,oeBetas,oeRidge);
    timings.ClosedTraining=toc(stageTimer);stageTimer=tic;
    [cs,cr]=bb_closed_report(closed,tr,va,te,cfg);summary=[summary;cs];
    timings.Evaluation=timings.Evaluation+toc(stageTimer);
    selection.Closed=closed.selected;selection.closedAccepted=closed.accepted;
    reports.ClosedLoop=cr;
    reports.ClosedSelection=closedSelection;
    save(fullfile(outDir,'closed_models.mat'),'closed','closedSelection','cfg','-v7');
end
stageTimer=tic;
if ~isempty(oofReport)
    selection.LOFO_DDV_RMSE=sqrt(sum(oofReport.SSE)/sum(oofReport.Samples));
    reports.OOF_DDV=oofReport;
end
if cfg.reportLoo
    loo=bb_ram_loo(tr,cfg,rates,chosenDDV);
    if ~isempty(loo)
        selection.loo=loo;reports.LOFO_RAM=loo;
    end
end
timings.Diagnostics=timings.Diagnostics+toc(stageTimer);stageTimer=tic;
if cfg.ddvFilterBank,reports.DDVPerformance=bb_ddv_report(ddvModels,tr,va,te,cfg);end
timings.Evaluation=timings.Evaluation+toc(stageTimer);stageTimer=tic;
compact=bb_print_summary(summary,selection,outDir,cfg);
audit=bb_data_diagnostics(tr,va,te,cfg);
if cfg.diagnostics||cfg.nlDiagnostics
    alignment=bb_alignment_report(tr,va,te,cfg);
    reports.Alignment=alignment;
end
eventConfigs=configs;wn=fieldnames(weightedRates);
for j=1:numel(wn)
 eventConfigs.(wn{j})=struct('ddv',ddvModels.(chosenDDV),'rate',weightedRates.(wn{j}),'angle',angle);
end
events=bb_event_report(eventConfigs,tr,va,te,cfg);
timings.Diagnostics=timings.Diagnostics+toc(stageTimer);
reports.Compact=compact;
runInfo=table({cfg.version},{source},logical(quick),cfg.Fs,numel(tr),numel(va),numel(te),...
    'VariableNames',{'Version','Source','SmokeMode','Fs','TrainFlights','ValidationFlights','TestFlights'});
runInfo.RAMTrainInput={cfg.ramTrainInput};runInfo.OEMaxTrainRows=cfg.oeMaxTrainRows;
runInfo.HoldBaselines=cfg.holdBaselines;runInfo.WienerEnabled=cfg.nlWiener;
reports.RunInfo=runInfo;
reports.PerFlight=details;
reports.DataAudit=audit;
reports.EventErrors=events;
if ~isempty(oeChoices)
    oeTable=cell2table(oeChoices,'VariableNames',{'Beta','Lags','Ridge','ValidationRMSE','WorstFlightRMSE'});
    if cfg.nlFocus&&cfg.nlFIRReference,oeTable.ReferenceOnly=true(height(oeTable),1);end
    reports.OE_Selection=oeTable;
end
reports.Summary=summary;
if ~isempty(nlReport),reports.NL_Selection=nlReport;end
reports.Split=manifest;
reports.ValidationSelection=cell2table(choices,'VariableNames',{'Stage','Label','Beta','Calibrated','Gain','Bias','ValidationRMSE'});
reports.SeedSensitivity=cell2table(seedRows,'VariableNames',{'Seed','ValidationRAMRMSE'});
save(fullfile(outDir,'trained_models.mat'),'configs','selection','ddvModels','cfg','manifest','-v7');
if cfg.nlFocus&&cfg.nlRefitTrainVal
    save(fullfile(outDir,'trained_models.mat'),'refitModels','-append');
end
stageTimer=tic;
if cfg.makePlots
    closedPlot=[];if strcmp(mode,'closed'),closedPlot=closed;end
    bb_plot_flight_outputs(flights,configs.(chosenRate),selection,outDir,closedPlot);
end
timings.Plots=toc(stageTimer);
% Workbook write cost is measured once; final timing stamp is explicitly excluded.
stageTimer=tic;book=fullfile(outDir,'comparison.xlsx');
layout=bb_write_workbook(book,reports,selection,cfg,timings);
timings.ReportWrite=toc(stageTimer);timings.Total=toc(runTimer);selection.timings=timings;
bb_stamp_timings(book,layout,timings,cfg);
save(fullfile(outDir,'selection.mat'),'selection','choices','oeChoices','cfg','timings','summary','layout','-v7');
save(fullfile(outDir,'trained_models.mat'),'selection','timings','-append');
fprintf('Stage times (seconds; final stamp/save excluded):\n');
names=fieldnames(timings);
for j=1:numel(names),fprintf('  %-18s %10.3f\n',names{j},timings.(names{j}));end
fprintf('Results: %s\n',outDir);
end
