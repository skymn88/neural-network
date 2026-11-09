function model=bb_rate_train(flights,cfg,kind,beta)
% Train residual velocity target from TRAIN observations only.
% dx/dt = beta*x + learned_function(DDV input history).
if beta>0 || 1+beta/cfg.Fs<=0,error('Invalid beta');end
for f=1:numel(flights)
    x=flights(f).S.RAM;
    previous=[x(1);x(1:end-1)];
    flights(f).S.RAMRate=(x-previous)*cfg.Fs-beta*previous;
end
inputs={'DDV'};
if cfg.usePressure,inputs=[inputs {'SupplyPressure','ReturnPressure'}];end
model=bb_fit_candidate(flights,cfg,[kind '_rate'],inputs,{'RAMRate'},kind);
model.beta=beta;model.stateKind='leaky_integrator';
end
