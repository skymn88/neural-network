/* Test double only. Not MathWorks headers and not a MATLAB/Simulink runtime. */
#ifndef TEST_SIMSTRUC_H
#define TEST_SIMSTRUC_H
#include <math.h>
#include <stddef.h>
typedef double real_T;typedef int int_T;
typedef struct {double value;int count,complex,sparse,type;} mxArray;
typedef struct {double work[8192],u,y,ts,offset;int expected,provided,width,dstate;const char *error;mxArray param;} SimStruct;
#define SS_DOUBLE 0
#define SS_OPTION_EXCEPTION_FREE_CODE 1
#define ssSetNumSFcnParams(S,n) ((S)->expected=(n))
#define ssGetNumSFcnParams(S) ((S)->expected)
#define ssGetSFcnParamsCount(S) ((S)->provided)
#define ssSetNumContStates(S,n) ((void)(S))
#define ssSetNumDiscStates(S,n) ((void)(S))
#define ssSetNumInputPorts(S,n) (1)
#define ssSetInputPortWidth(S,p,n) ((void)(S))
#define ssSetInputPortDataType(S,p,n) ((void)(S))
#define ssSetInputPortDirectFeedThrough(S,p,n) ((void)(S))
#define ssSetInputPortRequiredContiguous(S,p,n) ((void)(S))
#define ssSetNumOutputPorts(S,n) (1)
#define ssSetOutputPortWidth(S,p,n) ((void)(S))
#define ssSetOutputPortDataType(S,p,n) ((void)(S))
#define ssSetNumSampleTimes(S,n) ((void)(S))
#define ssSetNumDWork(S,n) (1)
#define ssSetDWorkWidth(S,p,n) ((S)->width=(n))
#define ssSetDWorkDataType(S,p,n) ((void)(S))
#define ssSetDWorkUsedAsDState(S,p,n) ((S)->dstate=(n))
#define ssSetOptions(S,n) ((void)(S))
#define ssSetSampleTime(S,p,n) ((S)->ts=(n))
#define ssSetOffsetTime(S,p,n) ((S)->offset=(n))
#define ssGetDWork(S,p) ((S)->work)
#define ssGetInputPortSignal(S,p) (&(S)->u)
#define ssGetOutputPortRealSignal(S,p) (&(S)->y)
#define ssGetSFcnParam(S,p) (&(S)->param)
#define ssSetSFcnParamTunable(S,p,n) ((void)(S))
#define ssGetErrorStatus(S) ((S)->error)
#define ssSetErrorStatus(S,msg) ((S)->error=(msg))
#define mxIsDouble(p) ((p)->type)
#define mxIsComplex(p) ((p)->complex)
#define mxIsSparse(p) ((p)->sparse)
#define mxGetNumberOfElements(p) ((p)->count)
#define mxGetScalar(p) ((p)->value)
#define mxIsFinite(x) isfinite(x)
#endif
