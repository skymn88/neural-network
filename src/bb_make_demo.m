function Raw = bb_make_demo()
% Synthetic smoke-test data only; NOT a model fidelity benchmark.
rng(123);
for f=1:2
    t=(0:1/64:55)'; n=numel(t);
    coil=0.3*sin(2*pi*(0.13+0.02*f)*t)+0.25*(t>12)-0.18*(t>33);
    coil=coil+0.02*randn(n,1);
    ddv=zeros(n,1); ram=zeros(n,1);
    ddv(1)=0.02*f;ram(1)=0.04*f;
    for k=2:n
        ddv(k)=ddv(k-1)+(1/64)*(1.4*coil(k)-ddv(k-1))/0.10;
        ram(k)=ram(k-1)+(1/64)*(0.2*ddv(k)-0.05*ram(k-1));
    end
    force=420*ddv+30*ram;
    angle=12*ram;
    Raw(f).name=sprintf('SYNTHETIC_%02d',f); %#ok<AGROW>
    Raw(f).CoilCurrent=bb_signal(t,coil);
    Raw(f).DDV=bb_signal(t,ddv);
    Raw(f).RAM=bb_signal(t,ram);
    Raw(f).ActuatorForce=bb_signal(t,force);
    Raw(f).Deflection=bb_signal(t,angle);
    Raw(f).SupplyPressure=bb_signal(t(1:2:end),3e7+1e5*sin(t(1:2:end)));
    Raw(f).ReturnPressure=bb_signal(t(1:2:end),1e6+2e4*cos(t(1:2:end)));
end
end
