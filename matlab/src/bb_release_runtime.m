function bb_release_runtime()
% User-requested cleanup, including error exits. Does not close ordinary figures.
if exist('bdclose','file')~=0
 try,bdclose('all');catch problem,warning('bb:CloseModels','bdclose failed: %s',problem.message);end
end
try,evalin('base','clear mex');catch problem,warning('bb:ClearMex','clear mex failed: %s',problem.message);end
end
