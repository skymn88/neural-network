/* Level-1 S-function: coil current -> DDV. Inputs: 1 (current). Outputs: 1 (DDV). */
#define S_FUNCTION_NAME sfun_ddv
#define S_FUNCTION_LEVEL 1
#include "simstruc.h"
#include "actuator_core.h"
#define NRW ((int)(sizeof(DdvState) / sizeof(double)))

static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 0);
    if (!ssSetNumInputPorts(S, 1)) return; ssSetInputPortWidth(S, 0, 1); ssSetInputPortDirectFeedThrough(S, 0, 1);
    if (!ssSetNumOutputPorts(S, 1)) return; ssSetOutputPortWidth(S, 0, 1);
    ssSetNumSampleTimes(S, 1); ssSetNumRWork(S, NRW);
}
static void mdlInitializeSampleTimes(SimStruct *S) { ssSetSampleTime(S, 0, 1.0 / FS); ssSetOffsetTime(S, 0, 0.0); }
#define MDL_INITIALIZE_CONDITIONS
static void mdlInitializeConditions(SimStruct *S) { int i; real_T *w = ssGetRWork(S); for (i = 0; i < NRW; i++) w[i] = 0.0; }
static void mdlOutputs(SimStruct *S, int_T tid)
{
    DdvState tmp; int i; real_T *w = ssGetRWork(S); double *p = (double *)&tmp;
    InputRealPtrsType u = ssGetInputPortRealSignalPtrs(S, 0);
    for (i = 0; i < NRW; i++) p[i] = w[i];
    ssGetOutputPortRealSignal(S, 0)[0] = ddv_step(&tmp, *u[0]);
}
#define MDL_UPDATE
static void mdlUpdate(SimStruct *S, int_T tid)
{
    DdvState tmp; int i; real_T *w = ssGetRWork(S); double *p = (double *)&tmp;
    InputRealPtrsType u = ssGetInputPortRealSignalPtrs(S, 0);
    for (i = 0; i < NRW; i++) p[i] = w[i];
    ddv_step(&tmp, *u[0]);
    for (i = 0; i < NRW; i++) w[i] = p[i];
}
static void mdlTerminate(SimStruct *S) {}
#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
