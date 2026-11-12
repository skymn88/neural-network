% Independent SCRIPT: run from the MATLAB editor; no training, no functions.
% Optional: paste a particular results folder below. Empty = newest USER run.
resultsFolder = '';
projectRoot=fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);setup_project;
if isempty(resultsFolder)
 candidates=dir(fullfile(projectRoot,'results','**','trained_models.mat'));
 [~,order]=sort([candidates.datenum],'descend');
 for q=order
  info=load(fullfile(candidates(q).folder,candidates(q).name),'selection');
  if isfield(info,'selection')&&isfield(info.selection,'source')&&strcmp(info.selection.source,'user')
   resultsFolder=candidates(q).folder;break;
  end
 end
end
assert(~isempty(resultsFolder),'No saved user run found. Set resultsFolder at top of this script.');
saved=load(fullfile(resultsFolder,'trained_models.mat'));
assert(strcmp(saved.selection.source,'user'),'This script expects a user-data run.');
rawFlights=load_user_flights();
plant=saved.configs.(saved.selection.RAM);
fprintf('Plot only, no training: %s | %s\n',resultsFolder,saved.selection.RAM);
% Edit ONLY this loop: 1 = first flight; [2 5] = two flights;
% 1:numel(rawFlights) = all flights. Keep the original data order.
for i=1
 flight=bb_prepare(rawFlights(i),saved.cfg);
 % Pass one flight so the existing three-panel plotting function is reused.
 bb_plot_flight_outputs(flight,plant,saved.selection,'');
end
