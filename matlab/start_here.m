% Open this script in MATLAB R2019a and press RUN (F5).
% Unit checks + supplied raw data + independent fixtures + smoke training.
projectDir=fileparts(mfilename('fullpath'));
addpath(projectDir,'-begin');setup_project();
checkResults=bb_run_checks();
