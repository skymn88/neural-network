function next=bb_ram_step(model,x,rate)
% Single-state un-clamped update, shared by open/closed/one-step inference.
next=(1+model.beta/model.Fs)*x+rate/model.Fs;
if isfield(model,'stateSpace')
 q=model.stateSpace;z=(x-q.center)/q.scale;v=tanh(rate/q.driveScale);
 % Bounded coefficients; linear state term can counter excessive base leakage.
 w=tanh(q.weights);correction=w(1)+w(2)*z+w(3)*tanh(z)+w(4)*v+w(5)*tanh(z).*v;
 next=next+q.scale*correction/model.Fs;
end
if isfield(model,'stateLeak')
 q=model.stateLeak;next=next-q.k*tanh((x-q.center)/q.scale)/model.Fs;
end
if isfield(model,'constantRAM'),next=zeros(size(next))+model.constantRAM;end
end
