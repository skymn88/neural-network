function figures=bb_plot_flight_outputs(flights,plant,selection,outDir,closed)
% Full-flight visualization, NOT a replacement for split-specific metrics.
% Change "for i=1" to "for i=1:numel(flights)" to plot every flight.
if nargin<5,closed=[];end
figures=[];
for i=1
 fl=flights(i);u=bb_predict_batch(plant.ddv,fl);
 ram=bb_rollout_rate(plant.rate,fl,fl.S.RAM(1),struct('DDV',u));
 label=['Plant ' selection.RAM];predictedCurrent=[];
 if ~isempty(closed)
  if strcmp(closed.selected,'Hybrid')
   [ram,info]=bb_closed_loop(closed.controller,closed.plant,fl,fl.S.RAM(1));predictedCurrent=info.current;
  else
   ram=bb_rollout_rate(closed.setpoint,fl,fl.S.RAM(1));
  end
  label=['Closed ' closed.selected];
 end
 angle=polyval(flipud(plant.angle.coefficients),ram/plant.angle.scale);
 h=figure('Name',[fl.name ' | ' label],'NumberTitle','off');figures=[figures h]; %#ok<AGROW>
 subplot(3,1,1);plot(fl.t,fl.S.CoilCurrent,'k');hold on;
 if isempty(predictedCurrent)
  legend('Measured current (input)','Location','best');
 else
  plot(fl.t,predictedCurrent,'r--');legend('Measured current','Predicted current','Location','best');
 end
 ylabel('Current');grid on;
 title([fl.name ' | full-flight diagnostic, not split score'],'Interpreter','none');
 subplot(3,1,2);plot(fl.t,fl.S.RAM,'k',fl.t,ram,'r--');ylabel('RAM');grid on;
 legend('Measured',label,'Location','best','Interpreter','none');
 subplot(3,1,3);plot(fl.t,fl.S.Deflection,'k',fl.t,angle,'r--');ylabel('Angle');xlabel('Time (s)');grid on;
 legend('Measured',label,'Location','best','Interpreter','none');
 if ~isempty(outDir)
  savefig(h,fullfile(outDir,sprintf('flight_%02d_outputs.fig',i)));
  print(h,fullfile(outDir,sprintf('flight_%02d_outputs.png',i)),'-dpng','-r120');
 end
end
end
