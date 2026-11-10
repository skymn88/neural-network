function outDir=run_rich_comparison(source,quick,mode,overrides)
% No NARX. Select on validation, then evaluate test. Bundled synthetic test is a reused regression case.
if nargin<1,source='synthetic';end
if nargin<2,quick=false;end
if nargin<3,mode='plant';end
if nargin<4,overrides=struct();end
if ~ismember(mode,{'plant','closed'}),error('Mode must be plant or closed');end
cfg=bb_config();names=fieldnames(overrides);
for j=1:numel(names)
    if ~isfield(cfg,names{j}),error('Unknown cfg field: %s',names{j});end
    cfg.(names{j})=overrides.(names{j});
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
mkdir(outDir);
labels={'MLP_basic','MLP_more_training','MLP_quasiNewton','FIR_rich','GP_coverage'};
if cfg.nlFocus,labels={'NL_OE'};end
ddvModels=struct();rates=struct();choices={};best=Inf;start=cfg.evalStartIndex;
for j=1:numel(labels)
    label=labels{j};[c,kind]=bb_rich_config(cfg,label);
    ddvModels.(label)=bb_fit_candidate(tr,c,label,{'CoilCurrent'},{'DDV'},kind);
    sse=0;n=0;
    for f=1:numel(va)
        p=bb_predict_batch(ddvModels.(label),va(f));idx=start:numel(p);
        sse=sse+sum((p(idx)-va(f).S.DDV(idx)).^2);n=n+numel(idx);
    end
    score=sqrt(sse/n);choices(end+1,:)={'DDV',label,NaN,false,1,0,score}; %#ok<AGROW>
    fprintf('DDV %s validation RMSE %.6g\n',label,score);
    if score<best,best=score;chosenDDV=label;end
end
if ~isfinite(best),error('All DDV validation candidates failed');end
vDDV=cell(size(va));for f=1:numel(va),vDDV{f}=bb_predict_batch(ddvModels.(chosenDDV),va(f));end
ramTr=tr;oofReport=table();
if ~ismember(cfg.ramTrainInput,{'measured','oof'}),error('Invalid ramTrainInput');end
if strcmp(cfg.ramTrainInput,'oof')||cfg.reportLoo
    [oofTr,oofReport]=bb_ddv_oof(tr,cfg,chosenDDV);
    if strcmp(cfg.ramTrainInput,'oof'),ramTr=oofTr;end
end
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
if ~cfg.nlFocus
for beta=oeBetas
 for lag=oeLags
  for ridge=oeRidge
    baseCfg=cfg;baseCfg.oeZeroRateAtNull=false;
    model=bb_oe_train(ramTr,baseCfg,beta,lag,ridge);model=bb_ram_bounds(model,tr,cfg);sse=0;n=0;worst=0;
    for f=1:numel(va)
        p=bb_rollout_rate(model,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));idx=start:numel(p);
        e=p(idx)-va(f).S.RAM(idx);sse=sse+sum(e.^2);n=n+numel(e);
        worst=max(worst,sqrt(mean(e.^2)));
    end
    score=sqrt(sse/n);oeChoices(end+1,:)={beta,lag,ridge,score,worst}; %#ok<AGROW>
    if isfinite(score)&&score<bestOE,bestOE=score;rates.FIR_OE=model;end
    if isfinite(score)&&score<bestAll,bestAll=score;chosenRate='FIR_OE';end
  end
 end
 fprintf('FIR_OE beta %.5g complete; best validation RMSE %.6g\n',beta,bestOE);
end
if ~isfield(rates,'FIR_OE'),error('No finite OE validation model');end
labels=[labels {'FIR_OE'}];
end
% Nonlinear finite-memory input map + position output-error fit (not NARX).
[rates.NL_OE,nlReport,nlBest]=bb_nl_search(ramTr,va,vDDV,cfg,oeLags,oeBetas,oeRidge);
if nlBest<bestAll,bestAll=nlBest;chosenRate='NL_OE';end
labels=[labels {'NL_OE'}];
if cfg.oeZeroRateAtNull&&~cfg.nlFocus
    zeroBest=Inf;
    for beta=oeBetas
     for lag=oeLags
      for ridge=oeRidge
       model=bb_oe_train(ramTr,cfg,beta,lag,ridge);model=bb_ram_bounds(model,tr,cfg);sse=0;n=0;
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
if ~isempty(extraReport),writetable(extraReport,fullfile(outDir,'comparison.xlsx'),'Sheet','StructuralSelection');end
seedRows={cfg.seed,bestAll};

if ~isfinite(bestAll),error('All validation models failed');end
angle=bb_angle_select(tr,va,start);
selection=struct('DDV',chosenDDV,'RAM',chosenRate,'criterion','validation cascade RAM RMSE','seed',cfg.seed,'smokeMode',logical(quick),'source',source);
selectedRate=rates.(chosenRate);selection.RAMBeta=selectedRate.beta;
selection.RAMLags=selectedRate.inputLags;
if isfield(selectedRate,'ridge'),selection.RAMRidge=selectedRate.ridge;end
if isfield(selectedRate,'tanhScale'),selection.RAMTanhScale=selectedRate.tanhScale;end
% No test outputs were used above.
configs=struct();
for j=1:numel(labels)
    label=labels{j};configs.(label)=struct('ddv',ddvModels.(chosenDDV),'rate',rates.(label),'angle',angle);
end
configs.HoldInitial=struct('ddv',ddvModels.(chosenDDV),'rate',bb_hold_model(cfg),'angle',angle);


[details,summary]=bb_evaluate_rich(configs,tr,va,te,cfg);
if strcmp(mode,'closed')
    [closed,closedSelection]=bb_closed_train(tr,va,configs.(chosenRate),cfg,oeLags,oeBetas,oeRidge);
    [cs,cr]=bb_closed_report(closed,tr,va,te,cfg);summary=[summary;cs];
    selection.Closed=closed.selected;selection.closedAccepted=closed.accepted;
    writetable(cr,fullfile(outDir,'comparison.xlsx'),'Sheet','ClosedLoop');
    writetable(closedSelection,fullfile(outDir,'comparison.xlsx'),'Sheet','ClosedSelection');
    save(fullfile(outDir,'closed_models.mat'),'closed','closedSelection','cfg','-v7');
end
if ~isempty(oofReport)
    selection.LOFO_DDV_RMSE=sqrt(sum(oofReport.SSE)/sum(oofReport.Samples));
    writetable(oofReport,fullfile(outDir,'comparison.xlsx'),'Sheet','OOF_DDV');
end
if cfg.reportLoo
    loo=bb_ram_loo(tr,cfg,rates,chosenDDV);
    if ~isempty(loo)
        selection.loo=loo;writetable(loo,fullfile(outDir,'comparison.xlsx'),'Sheet','LOFO_RAM');
    end
end
compact=bb_print_summary(summary,selection,outDir,cfg);
audit=bb_data_diagnostics(tr,va,te,cfg);
if cfg.diagnostics
    alignment=bb_alignment_report(tr,va,te,cfg);
    writetable(alignment,fullfile(outDir,'comparison.xlsx'),'Sheet','Alignment');
end
events=bb_event_report(configs,tr,va,te,cfg);
writetable(compact,fullfile(outDir,'comparison.xlsx'),'Sheet','Compact');
runInfo=table({cfg.version},{source},logical(quick),cfg.Fs,numel(tr),numel(va),numel(te),...
    'VariableNames',{'Version','Source','SmokeMode','Fs','TrainFlights','ValidationFlights','TestFlights'});
writetable(runInfo,fullfile(outDir,'comparison.xlsx'),'Sheet','RunInfo');
writetable(details,fullfile(outDir,'comparison.xlsx'),'Sheet','PerFlight');
writetable(audit,fullfile(outDir,'comparison.xlsx'),'Sheet','DataAudit');
writetable(events,fullfile(outDir,'comparison.xlsx'),'Sheet','EventErrors');
if ~isempty(oeChoices)
    writetable(cell2table(oeChoices,'VariableNames',{'Beta','Lags','Ridge','ValidationRMSE','WorstFlightRMSE'}),fullfile(outDir,'comparison.xlsx'),'Sheet','OE_Selection');
end
writetable(summary,fullfile(outDir,'comparison.xlsx'),'Sheet','Summary');
writetable(nlReport,fullfile(outDir,'comparison.xlsx'),'Sheet','NL_Selection');
writetable(manifest,fullfile(outDir,'comparison.xlsx'),'Sheet','Split');
writetable(cell2table(choices,'VariableNames',{'Stage','Label','Beta','Calibrated','Gain','Bias','ValidationRMSE'}),fullfile(outDir,'comparison.xlsx'),'Sheet','ValidationSelection');
writetable(cell2table(seedRows,'VariableNames',{'Seed','ValidationRAMRMSE'}),fullfile(outDir,'comparison.xlsx'),'Sheet','SeedSensitivity');
save(fullfile(outDir,'selection.mat'),'selection','choices','oeChoices','cfg','-v7');
save(fullfile(outDir,'trained_models.mat'),'configs','selection','ddvModels','cfg','manifest','-v7');
if cfg.makePlots
    closedPlot=[];if strcmp(mode,'closed'),closedPlot=closed;end
    bb_plot_flight_outputs(flights,configs.(chosenRate),selection,outDir,closedPlot);
end
fprintf('Results: %s\n',outDir);
end
