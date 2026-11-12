function root=setup_project()
% Add only active project folders. Does not recursively add data or results.
root=fileparts(mfilename('fullpath'));
addpath(root,'-begin');addpath(fullfile(root,'src'),'-begin');addpath(fullfile(root,'tests'),'-begin');
end
