function xFiltered = bb_antialias(t,x,targetFs,signalName)
% Base-MATLAB FIR anti-aliasing before reducing sample rate.
% Near-uniform native sampling is expected. No Signal Processing Toolbox.
xFiltered = x;
dt = diff(t);
fsNative = 1/median(dt);
if fsNative <= 1.05*targetFs
    return;
end
jitter = max(abs(dt-median(dt)))/median(dt);
if jitter > 0.10
    warning('bb:irregularSampling', ...
        '%s: irregular high-rate signal: FIR skipped. Check timestamps / aliasing.',signalName);
    return;
end
% Low-pass cutoff 0.44 * target sample rate (Nyquist target = 0.5 Fs).
fc = 0.44*targetFs/fsNative; % cycles per native sample
half = min(128,max(20,ceil(4*fsNative/targetFs)));
n = (-half:half)';
z = 2*fc*n;
sincz = ones(size(z));
nonzero = abs(z) > 1e-12;
sincz(nonzero) = sin(pi*z(nonzero))./(pi*z(nonzero));
h = (2*fc).*sincz;
w = 0.54 - 0.46*cos(2*pi*(0:numel(n)-1)'/(numel(n)-1));
h = h.*w;
h = h/sum(h);
padded = [repmat(x(1),half,1);x;repmat(x(end),half,1)];
xFiltered = conv(padded,h,'valid');
end
