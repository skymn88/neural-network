function report=bb_data_diagnostics(tr,va,te,cfg)
% Train-only reference ranges. Diagnostic only: never changes split/model data.
channels={'CoilCurrent','DDV','RAM','Deflection'};
if cfg.usePressure,channels=[channels {'SupplyPressure','ReturnPressure'}];end
sets={tr,va,te};splits={'train','validation','test'};rows={};
for j=1:numel(channels)
    name=channels{j};train=[];
    for f=1:numel(tr),train=[train;tr(f).S.(name)];end %#ok<AGROW>
    lo=min(train);hi=max(train);mu=mean(train);sigma=std(train);
    for s=1:3
        fs=sets{s};
        for f=1:numel(fs)
            x=fs(f).S.(name);outside=100*mean(x<lo|x>hi);
            shift=NaN;if sigma>1e-12,shift=(mean(x)-mu)/sigma;end
            dx=diff(x)*cfg.Fs;
            rows(end+1,:)={fs(f).name,splits{s},name,numel(x),min(x),max(x),mean(x),std(x),outside,shift,sqrt(mean(dx.^2)),x(1),x(end)}; %#ok<AGROW>
        end
    end
end
report=cell2table(rows,'VariableNames',{'Flight','Split','Signal','Samples','Min','Max','Mean','Std','OutsideTrainRangePct','MeanShiftTrainStd','DifferenceRMSperSecond','Initial','Final'});
end
