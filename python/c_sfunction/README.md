# Level-1 C S-functions (null_rate chain, 64 Hz)
Three blocks in series: `sfun_ddv` (current -> DDV), `sfun_ram` (DDV -> RAM, dialog parameter x0), `sfun_angle` (RAM -> angle).
1. `python gen_params.py ../audit_results/<run>/models.json <model_name>` -> writes `actuator_params.h`
2. In MATLAB: `mex sfun_ddv.c`, `mex sfun_ram.c`, `mex sfun_angle.c`; use S-Function blocks, sample time 1/64 s.
3. `python run_parity.py <models.json>` checks the C core (`actuator_core.h`) against Python (gcc, ~1e-14).
Supports ram.kind=null_rate with ddv.basis=rich (legacy/stationary/envelope observers). Simulink build itself is untested here.
