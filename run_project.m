function resultsDir=run_project(source,mode,overrides)
% Project training entry point. Explicit user data path preserved.
% run_project() uses load_user_flights; run_project('synthetic') uses fixture.
if nargin<1,source='user';end
if nargin<2,mode='plant';end
if nargin<3,overrides=struct();end
setup_project();
resultsDir=run_rich_comparison(source,false,mode,overrides);
end
