function resultsDir=run_project(source)
% Project training entry point. Explicit user data path preserved.
% run_project() uses load_user_flights; run_project('synthetic') uses fixture.
if nargin<1,source='user';end
setup_project();
resultsDir=run_rich_comparison(source);
end
