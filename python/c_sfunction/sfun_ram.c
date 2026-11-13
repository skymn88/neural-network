/* Level-1 S-function: coil DDV -> RAM position. Parameter 1: x0 (initial RAM position). */
#define S_FUNCTION_NAME sfun_ram
#define S_FUNCTION_LEVEL 1
#include "simstruc.h"
#include "actuator_core.h"
#define X0 (*mxGetPr(ssGetSFcnParam(S, 0)))
#define NRW ((int)(sizeof(RamState) / sizeof(double)))

static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 1);
    if (!ssSetNumInputPorts(S, 1)) return; ssSetInputPortWidth(S, 0, 1); ssSetInputPortDirectFeedThrough(S, 0, 1);
    if (!ssSetNumOutputPorts(S, 1)) return; ssSetOutputPortWidth(S, 0, 1);
    ssSetNumSampleTimes(S, 1); ssSetNumRWork(S, NRW);
}
static void mdlInitializeSampleTimes(SimStruct *S) { ssSetSampleTime(S, 0, 1.0 / FS); ssSetOffsetTime(S, 0, 0.0); }
#define MDL_INITIALIZE_CONDITIONS
static void mdlInitializeConditions(SimStruct *S) { int i; real_T *w = ssGetRWork(S); for (i = 0; i < NRW; i++) w[i] = 0.0; /* init flag 0 => ram_step sets x=x0 */ }
static void mdlOutputs(SimStruct *S, int_T tid)
{
    RamState tmp; int i; real_T *w = ssGetRWork(S); double *p = (double *)&tmp;
    InputRealPtrsType u = ssGetInputPortRealSignalPtrs(S, 0);
    for (i = 0; i < NRW; i++) p[i] = w[i];
    ssGetOutputPortRealSignal(S, 0)[0] = ram_step(&tmp, *u[0], X0);
}
#define MDL_UPDATE
static void mdlUpdate(SimStruct *S, int_T tid)
{
    RamState tmp; int i; real_T *w = ssGetRWork(S); double *p = (double *)&tmp;
    InputRealPtrsType u = ssGetInputPortRealSignalPtrs(S, 0);
    for (i = 0; i < NRW; i++) p[i] = w[i];
    ram_step(&tmp, *u[0], X0);
    for (i = 0; i < NRW; i++) w[i] = p[i];
}
static void mdlTerminate(SimStruct *S) {}
#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
