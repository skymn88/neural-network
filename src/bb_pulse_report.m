function report=bb_pulse_report(configs,tr,cfg,outDir)
% Common TRAIN-scaled synthetic stimulus, NOT real-flight validation or a target.
fl=bb_pulse_flight(tr,cfg);names=fieldnames(configs);report=table();fig=[];
if cfg.makePlots,fig=figure('Name','Synthetic PulseHold diagnostic','NumberTitle','off');end
for j=1:numel(names)
 plant=configs.(names{j});ddv=bb_predict_batch(plant.ddv,fl);
 ram=bb_rollout_rate(plant.rate,fl,fl.S.RAM(1),struct('DDV',ddv));
 angle=polyval(flipud(plant.angle.coefficients),ram/plant.angle.scale);
 values={ram,angle};stages={'RAM','Angle'};
 for k=1:2
  tab=bb_pulse_metrics(fl,values{k},cfg.Fs);
  tab.Model=repmat(names(j),height(tab),1);tab.Stage=repmat(stages(k),height(tab),1);
  report=[report;tab]; %#ok<AGROW>
 end
 if ~isempty(fig)
  signals={ddv,ram,angle};labels={'DDV','RAM','Angle'};
  for k=1:3
   subplot(3,1,k);plot(fl.t,signals{k},'DisplayName',names{j});hold on;grid on;ylabel(labels{k});
  end
 end
end
if ~isempty(fig)
 for k=1:3,subplot(3,1,k);legend('show','Location','best','Interpreter','none');end
 xlabel('Time (s)');subplot(3,1,1);title('Synthetic input: diagnostic only, no measured reference');
 if ~isempty(outDir),print(fig,fullfile(outDir,'pulse_hold.png'),'-dpng','-r120');end
end
end
