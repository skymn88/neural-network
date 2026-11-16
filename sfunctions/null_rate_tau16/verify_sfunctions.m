function verify_sfunctions()
% Local MATLAB/Simulink check against the exported deterministic Python fixture.
root = fileparts(mfilename('fullpath'));
fixture = dlmread(fullfile(root,'parity_fixture.csv'),',',1,0);
name = ['actuator_check_' datestr(now,'HHMMSSFFF')];
load_system('simulink');
new_system(name);
cleanup = onCleanup(@() bdclose(name)); %#ok<NASGU>
set_param(name,'SolverType','Fixed-step','Solver','FixedStepDiscrete', ...
    'FixedStep','1/64','StopTime',num2str(fixture(end,1),17));
add_block('simulink/Sources/From Workspace',[name '/current'], ...
    'VariableName','current_data','Interpolate','off');
names = {'bb_curr_ddv','bb_ddv_ram','bb_ram_angle'};
for k = 1:3
    add_block('simulink/User-Defined Functions/S-Function',[name '/' names{k}], ...
        'FunctionName',names{k});
    if k == 2
        set_param([name '/' names{k}],'Parameters',num2str(fixture(1,4),17));
    end
    add_block('simulink/Sinks/To Workspace',[name '/output' num2str(k)], ...
        'VariableName',['y' num2str(k)],'SaveFormat','Timeseries');
    add_line(name,[names{k} '/1'],['output' num2str(k) '/1']);
    if k > 1
        add_line(name,[names{k-1} '/1'],[names{k} '/1']);
    end
end
add_line(name,'current/1','bb_curr_ddv/1');
input = Simulink.SimulationInput(name);
input = input.setVariable('current_data',fixture(:,1:2));
output = sim(input);
for k = 1:3
    series = output.get(['y' num2str(k)]);
    predicted = reshape(series.Data,[],1);
    assert(numel(predicted)==size(fixture,1),'Unexpected output sample count.');
    assert(max(abs(series.Time(:)-fixture(:,1)))<1e-10,'Unexpected output timing.');
    difference = max(abs(predicted-fixture(:,k+2)));
    fprintf('%s maximum absolute error: %.12g\n',names{k},difference);
    assert(difference<1e-7,'Python/MEX parity failed.');
end
fprintf('All three MEX blocks match the Python fixture at 64 Hz.\n');
end
