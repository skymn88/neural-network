/* Level-1 S-function: RAM position -> deflection angle (static polynomial). */
#define S_FUNCTION_NAME sfun_angle
#define S_FUNCTION_LEVEL 1
#include "simstruc.h"
#include "actuator_core.h"
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 0);
    if (!ssSetNumInputPorts(S, 1)) return; ssSetInputPortWidth(S, 0, 1); ssSetInputPortDirectFeedThrough(S, 0, 1);
    if (!ssSetNumOutputPorts(S, 1)) return; ssSetOutputPortWidth(S, 0, 1);
    ssSetNumSampleTimes(S, 1);
}
static void mdlInitializeSampleTimes(SimStruct *S) { ssSetSampleTime(S, 0, 1.0 / FS); ssSetOffsetTime(S, 0, 0.0); }
static void mdlOutputs(SimStruct *S, int_T tid)
{
    InputRealPtrsType u = ssGetInputPortRealSignalPtrs(S, 0);
    ssGetOutputPortRealSignal(S, 0)[0] = angle_eval(*u[0]);
}
static void mdlTerminate(SimStruct *S) {}
#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
