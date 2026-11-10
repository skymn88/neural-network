function next=bb_ram_step(model,x,rate)
% Single-state un-clamped update, shared by open/closed/one-step inference.
next=(1+model.beta/model.Fs)*x+rate/model.Fs;
if isfield(model,'stateLeak')
 q=model.stateLeak;next=next-q.k*tanh((x-q.center)/q.scale)/model.Fs;
end
if isfield(model,'constantRAM'),next=zeros(size(next))+model.constantRAM;end
end
