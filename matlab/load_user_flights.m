function Raw=load_user_flights()
% Stable adapter entry point. Keep private paths/mappings in the ignored file.
if exist('load_user_flights_local','file')~=2
    error(['Create load_user_flights_local.m in the matlab folder from ' ...
        'examples/load_user_flights_local.m and map your measured columns.']);
end
Raw=load_user_flights_local();
if isempty(Raw)||~isstruct(Raw),error('The local adapter must return a nonempty Raw struct array');end
end
