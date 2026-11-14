#ifndef ACTUATOR_CORE_H
#define ACTUATOR_CORE_H
#include <stddef.h>
#define DDV_HISTORY 17
#define DDV_FEATURES 154
#define DDV_RICH 1
#define MAX_HIDDEN 32
#define MEMBER_COUNT 3
#define QUIET_WINDOW 32
#define REF_DELAY 1
#define REF_KNOTS 81
#define ANGLE_COUNT 4

typedef struct {double ready,u[DDV_HISTORY];} BBDdvState;
typedef struct {double ready,x,center,previous,was_quiet;double ddv[QUIET_WINDOW],delay[REF_DELAY+1];double hidden[MEMBER_COUNT][MAX_HIDDEN],cell[MEMBER_COUNT][MAX_HIDDEN];} BBRamState;
size_t bb_ddv_state_size(void);size_t bb_ram_state_size(void);
void bb_ddv_init(BBDdvState *s);double bb_ddv_step(BBDdvState *s,double current);
void bb_ram_init(BBRamState *s,double x0);double bb_ram_step(BBRamState *s,double ddv);
double bb_angle(double ram);
#endif
