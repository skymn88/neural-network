/* gcc parity harness: reads "current ddv ram angle" rows, reports max abs error. x0 from argv[2]. */
#include <stdio.h>
#include <stdlib.h>
#include "actuator_core.h"
int main(int c, char **v)
{
    FILE *f = fopen(v[1], "r"); double x0 = atof(v[2]), u, d, r, a, e1 = 0, e2 = 0, e3 = 0;
    static DdvState ds; static RamState rs; long n = 0;
    while (fscanf(f, "%lf %lf %lf %lf", &u, &d, &r, &a) == 4) {
        double yd = ddv_step(&ds, u), yr = ram_step(&rs, d, x0), ya = angle_eval(r);
        if (fabs(yd - d) > e1) e1 = fabs(yd - d);
        if (fabs(yr - r) > e2) e2 = fabs(yr - r);
        if (fabs(ya - a) > e3) e3 = fabs(ya - a);
        n++;
    }
    printf("n=%ld max|ddv|=%.3e max|ram|=%.3e max|angle|=%.3e\n", n, e1, e2, e3);
    return 0;
}
