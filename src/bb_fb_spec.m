function spec=bb_fb_spec(cfg)
% Fixed stable filters, independent of any observed flight/output statistics.
if any(abs(cfg.fbPoles)>=1)||any(~isfinite(cfg.fbPoles)),error('Unstable FB poles');end
if any(~isfinite(cfg.fbFrequencies))||any(~isfinite(cfg.fbDamping))||any(cfg.fbFrequencies<=0|cfg.fbFrequencies>=cfg.Fs/2)||any(cfg.fbDamping<=0|cfg.fbDamping>=1),error('Invalid resonance settings');end
if cfg.fbRawLags<0||cfg.fbRawLags~=floor(cfg.fbRawLags),error('Invalid raw lag count');end
spec=struct('Fs',cfg.Fs,'poles',cfg.fbPoles(:)','rawLags',cfg.fbRawLags,'a',zeros(0,3),'b',zeros(0,1));
for f=cfg.fbFrequencies(:)'
 for d=cfg.fbDamping(:)'
  radius=exp(-d*2*pi*f/cfg.Fs);theta=2*pi*f*sqrt(1-d*d)/cfg.Fs;
  den=[1 -2*radius*cos(theta) radius^2];spec.a(end+1,:)=den;spec.b(end+1,1)=sum(den);
 end
end
end
