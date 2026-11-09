function flight = bb_prepare(raw,cfg)
% Align only selected model channels to a common 64-Hz clock.
% Note: 1-Hz command signals are not used in the physical cascade.
required = {'CoilCurrent','DDV','RAM','Deflection'};
if ~isfield(cfg,'requireForce') || cfg.requireForce
    required=[required {'ActuatorForce'}];
end
if cfg.usePressure
    required = [required {'SupplyPressure','ReturnPressure'}];
end
clean = struct(); t0 = -inf; t1 = inf;
for i = 1:numel(required)
    name = required{i};
    if ~isfield(raw,name)
        error('Missing required channel %s in load_user_flights.m.',name);
    end
    s = raw.(name);
    if ~isstruct(s) || ~isfield(s,'t') || ~isfield(s,'x')
        error('%s must be supplied using bb_signal(time,values).',name);
    end
    t = double(s.t(:)); x = double(s.x(:));
    if numel(t) ~= numel(x), error('%s: time/data mismatch',name); end
    valid = isfinite(t) & isfinite(x);
    t=t(valid); x=x(valid);
    if numel(t)<3, error('%s: fewer than 3 valid samples',name); end
    [t,ord] = sort(t); x=x(ord);
    [t,ord] = unique(t,'last'); x=x(ord);
    if numel(t)<3, error('%s: fewer than 3 unique times',name); end
    dt = diff(t); dtMid = median(dt);
    if any(dt<=0), error('%s: time axis not strictly increasing',name); end
    if max(dt) > cfg.maxGapFactor*dtMid
        error('%s: large time gaps (max %.5g vs median %.5g sec).',name,max(dt),dtMid);
    end
    clean.(name) = struct('t',t,'x',x,'nativeHz',1/dtMid);
    t0=max(t0,t(1)); t1=min(t1,t(end));
end
if t1<=t0, error('No common time interval across selected signals.'); end
N=floor((t1-t0)*cfg.Fs)+1;
if N<=3*(max(cfg.inputLags,cfg.outputLags)+3)
    error('Flight too short for training / lag window.');
end
tgrid=t0+(0:N-1)'/cfg.Fs;
flight.name='Flight';
if isfield(raw,'name'), flight.name=char(raw.name); end
flight.t=tgrid-tgrid(1); flight.S=struct(); flight.Fs=cfg.Fs;
for i=1:numel(required)
    name=required{i}; s=clean.(name);
    xf=bb_antialias(s.t,s.x,cfg.Fs,name);
    flight.S.(name)=interp1(s.t,xf,tgrid,'linear');
    if any(~isfinite(flight.S.(name)))
        error('%s: resampling resulted in NaN/Inf',name);
    end
end
end
