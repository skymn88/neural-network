function build_sfunctions()
% Windows x64 MATLAB R2019a + Simulink; no network/package installation.
if ~ispc || ~strcmp(mexext,'mexw64')
    error('actuator:platform','Run this build in Windows x64 MATLAB to create .mexw64.');
end
root = fileparts(mfilename('fullpath'));
old = pwd;
cleanup = onCleanup(@() cd(old)); %#ok<NASGU>
cd(root);
names = {'bb_curr_ddv','bb_ddv_ram','bb_ram_angle'};
for k = 1:numel(names)
    clear(names{k});
    mex('-R2018a', '-largeArrayDims', ['-I' fullfile(matlabroot,'simulink','include')], ...
        [names{k} '.c'], 'actuator_core.c', '-output', names{k});
    assert(exist([names{k} '.' mexext],'file') == 3, 'MEX output missing.');
    fprintf('Built %s.%s\n',names{k},mexext);
end
fprintf('RAM block parameter: initial RAM position in physical units. Model sample time: 1/64 s.\n');
end
