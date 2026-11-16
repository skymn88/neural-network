# Generated C MEX S-Functions

Coefficients are embedded in `model_coefficients.h`, copied from the selected JSON model. No JSON/Python/PyTorch is required at inference time. Keep the core/header/coefficient files next to the three block sources.

On **Windows x64 MATLAB R2019a with Simulink and a supported C compiler**, change MATLAB Current Folder to this folder and run:

```matlab
build_sfunctions
verify_sfunctions
```

`verify_sfunctions` creates a temporary three-block model and checks its output against the shipped deterministic Python fixture; it then closes that temporary model. This MATLAB test has not been run in the Linux environment.

The first use may require `mex -setup C`. The command produces `bb_curr_ddv.mexw64`, `bb_ddv_ram.mexw64`, `bb_ram_angle.mexw64`. They are not precompiled in the Linux export environment.

Add three **S-Function** blocks with names `bb_curr_ddv`, `bb_ddv_ram`, `bb_ram_angle`; wire Current → DDV → RAM → Angle. The RAM block has **one parameter: initial measured RAM position**, in physical units (for example `-100`). Other blocks have no parameters. All ports are scalar real double and all outputs use original physical units. These are Level-2 C MEX S-Functions, not MATLAB Level-2 functions.

All three model blocks run at **1/64 s**. Do not change to 0.005 s: doing so changes learned dynamics. If the surrounding interface needs 200 Hz, use a Rate Transition/zero-order-hold boundary; it does not create new 200 Hz learned dynamics and its scheduling delay must be checked in the host model. No interpolation/look-ahead is used in the exported model.

Hidden/cell/null/delay/position state is per block instance in DWork; repeated mdlOutputs calls do not advance state. mdlUpdate commits one 64 Hz step; initialize/reset starts a new flight. No measured RAM is fed back after initial x0. This is a synthetic validation-selected research model, not real-flight qualification. C core parity is checked in Linux; MATLAB/Simulink execution and Windows MEX loading must be checked locally.
