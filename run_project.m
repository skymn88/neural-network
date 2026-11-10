function resultsDir=run_project(source,mode,overrides)
% SHORT COMMANDS (no setpoint needed for default plant):
%   run_project()                 % your flights, plant
%   run_project('user','closed')  % requires measured 64 Hz RAMSetpoint
% Optional settings: edit src/bb_config.m ONCE, then use the short commands.
% Default: RAM NL_FB, beta [-.3 0], OOF DDV, all OE samples, diagnostics.
% Optional: nlReference=true; nlWiener=true; oeTransitionWeight=[0 2 5].
% holdBaselines=true restores HoldInitial and TrainMean only when requested.
% Optional experiments: nlJointSearch=true; nlWideBeta=true;
% nlRefitTrainVal=true; nlStateLeak=true. All are OFF by default.
% One-off override example:
% run_project('user','plant',struct('betaCandidates',[0 -.1 -.3 -1 -2 -3]));
% run_project() uses load_user_flights; run_project('synthetic') uses fixture.
if nargin<1,source='user';end
if nargin<2,mode='plant';end
if nargin<3,overrides=struct();end
setup_project();
resultsDir=run_rich_comparison(source,false,mode,overrides);
end
