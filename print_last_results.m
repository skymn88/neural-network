% Print an existing v7/v8 report compactly WITHOUT retraining.
projectDir=fileparts(mfilename('fullpath'));addpath(projectDir,'-begin');setup_project();
[file,folder]=uigetfile('comparison.xlsx','Select an existing comparison.xlsx');
if ~isequal(file,0)
    report=readtable(fullfile(folder,file),'Sheet','Summary');
    selection=struct();
    if exist(fullfile(folder,'selection.mat'),'file')==2
        saved=load(fullfile(folder,'selection.mat'));selection=saved.selection;
    end
    if exist(fullfile(folder,'trained_models.mat'),'file')==2
        savedConfig=load(fullfile(folder,'trained_models.mat'),'cfg');
        if isfield(savedConfig,'cfg')&&isfield(savedConfig.cfg,'quick')
            selection.smokeMode=logical(savedConfig.cfg.quick);
        end
    end
    bb_print_summary(report,selection,folder);
end
