function s = bb_signal(t,x)
% Signal adapter: numeric time in seconds, duration, or datetime.
if isdatetime(t)
    t = posixtime(t);
elseif isduration(t)
    t = seconds(t);
end
if ~isnumeric(t) || ~isnumeric(x)
    error('Time and signal must be numeric (or time may be datetime/duration).');
end
s.t = double(t(:));
s.x = double(x(:));
if numel(s.t) ~= numel(s.x)
    error('bb_signal: time and data lengths differ.');
end
end
