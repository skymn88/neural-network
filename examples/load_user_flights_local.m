function Raw=load_user_flights_local()
% COPY this file to repository root as load_user_flights_local.m, then edit.
% Copy to the root before use; examples is not added to the MATLAB path.
root=fileparts(mfilename('fullpath'));
files={'flight01.csv','flight02.csv'}; % replace with your actual list
Raw=struct([]);
for k=1:numel(files)
    T=readtable(fullfile(root,'data','user',files{k}));
    Raw(k).name=sprintf('flight_%02d',k);
    % Column names below are placeholders. Each signal may use its own time.
    Raw(k).CoilCurrent=bb_signal(T.time_current,T.current);
    Raw(k).DDV=bb_signal(T.time_ddv,T.ddv);
    Raw(k).RAM=bb_signal(T.time_ram,T.ram);
    Raw(k).Deflection=bb_signal(T.time_angle,T.angle);
    % Closed mode / cfg.useSetpoint=true, measured 64 Hz channel only:
    % Raw(k).RAMSetpoint=bb_signal(T.time_ram_setpoint,T.ram_setpoint);
    % Only when cfg.usePressure=true:
    % Raw(k).SupplyPressure=bb_signal(T.time_supply,T.supply_pressure);
    % Raw(k).ReturnPressure=bb_signal(T.time_return,T.return_pressure);
end
end
