function [z,state]=bb_ctrl_features(e,ddvPrevious,currentPrevious,state,model)
% Causal controller: e(k)=setpoint(k)-RAMhat(k-1). No same-step algebraic loop.
L=model.lags;
if isempty(state)
 state.error=zeros(1,L);state.ddv=zeros(1,L);state.current=zeros(1,L);
 state.integral=zeros(1,numel(model.taus));
end
state.integral=exp(-1./(model.Fs*model.taus)).*state.integral+e/model.Fs;
state.ddv=[ddvPrevious state.ddv(1:end-1)];state.current=[currentPrevious state.current(1:end-1)];
z=[e state.error state.integral state.ddv state.current];
state.error=[e state.error(1:end-1)];
end
