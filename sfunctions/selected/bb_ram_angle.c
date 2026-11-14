/* One scalar physical input/output; discrete 64 Hz. State advances only in mdlUpdate. */
#define S_FUNCTION_NAME bb_ram_angle
#define S_FUNCTION_LEVEL 2
#include "simstruc.h"
#include "actuator_core.h"
#include <string.h>
#include <float.h>

static void mdlInitializeSizes(SimStruct *S){
    ssSetNumSFcnParams(S,0);
    if(ssGetNumSFcnParams(S)!=ssGetSFcnParamsCount(S))return;

    ssSetNumContStates(S,0);ssSetNumDiscStates(S,0);
    if(!ssSetNumInputPorts(S,1))return;
    ssSetInputPortWidth(S,0,1);ssSetInputPortDataType(S,0,SS_DOUBLE);
    ssSetInputPortDirectFeedThrough(S,0,1);ssSetInputPortRequiredContiguous(S,0,1);
    if(!ssSetNumOutputPorts(S,1))return;
    ssSetOutputPortWidth(S,0,1);ssSetOutputPortDataType(S,0,SS_DOUBLE);
    ssSetNumSampleTimes(S,1);
    if(!ssSetNumDWork(S,0))return;

    ssSetOptions(S,SS_OPTION_EXCEPTION_FREE_CODE);
}
static void mdlInitializeSampleTimes(SimStruct *S){ssSetSampleTime(S,0,1.0/64.0);ssSetOffsetTime(S,0,0.0);}
#define MDL_INITIALIZE_CONDITIONS
static void mdlInitializeConditions(SimStruct *S){(void)S;}
static void mdlOutputs(SimStruct *S,int_T tid){
    const real_T *u=(const real_T*)ssGetInputPortSignal(S,0);
    real_T *y=ssGetOutputPortRealSignal(S,0);(void)tid;
    if(!(u[0]>=-DBL_MAX && u[0]<=DBL_MAX)){ssSetErrorStatus(S,"Input must be a finite physical scalar.");return;}
    y[0]=bb_angle(u[0]);
}
#define MDL_UPDATE
static void mdlUpdate(SimStruct *S,int_T tid){
    (void)tid;
    if(ssGetErrorStatus(S)!=NULL)return;
    (void)S;
}
static void mdlTerminate(SimStruct *S){(void)S;}
#ifdef MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
