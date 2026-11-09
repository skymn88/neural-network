function paths=bb_paths()
% Centralized paths, independent of the MATLAB current working folder.
paths.root=fileparts(fileparts(mfilename('fullpath')));
paths.synthetic=fullfile(paths.root,'data','synthetic','synthetic_rich.mat');
paths.fixtures=fullfile(paths.root,'tests','fixtures');
paths.results=fullfile(paths.root,'results');
end
