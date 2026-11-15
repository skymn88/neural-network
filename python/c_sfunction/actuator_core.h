/* Pure-C core of the null_rate actuator chain (no Simulink dependency).
 * State structs contain doubles only, so they can live in S-function RWork.
 * Mirrors python/actuator/inference.py (rollout, neutral_features). */
#ifndef ACTUATOR_CORE_H
#define ACTUATOR_CORE_H
#include <math.h>
#include "actuator_params.h"

#define NH (DDV_LAGS + 1)
#define QWN ((OBS_QW) > 0 ? (OBS_QW) : 1)
#define DLN ((RAM_DELAY) > 0 ? (RAM_DELAY) : 1)

/* ---- Stage 1: current -> DDV ---- */
typedef struct { double init; double h[NH]; } DdvState;

static double ddv_step(DdvState *s, double u)
{
    int i, j; double acc, z, a;
    static const double ts[5] = {0.5, 1, 2, 4, 8}, ds[2] = {0.05, 0.15};
    if (s->init == 0.0) { for (i = 0; i < NH; i++) s->h[i] = u; s->init = 1.0; }
    else { for (i = NH - 1; i > 0; i--) s->h[i] = s->h[i - 1]; s->h[0] = u; }
    acc = DDV_W[0];
    for (j = 0; j < NH; j++) {
        z = s->h[j] / DDV_SCALE; a = fabs(z);
        acc += DDV_W[1 + j] * z + DDV_W[1 + NH + j] * a;
        for (i = 0; i < 5; i++) acc += DDV_W[1 + (2 + i) * NH + j] * tanh(z / ts[i]);
        for (i = 0; i < 2; i++)
            acc += DDV_W[1 + (7 + i) * NH + j] * (z > 0 ? 1.0 : (z < 0 ? -1.0 : 0.0)) * (a > ds[i] ? a - ds[i] : 0.0);
    }
    return acc;
}

/* ---- Stage 2: DDV -> RAM position ---- */
typedef struct {
    double init, k, x, estimate, ready, was_quiet, previous;
    double high, low, high_at, low_at, has_high, has_low, prev_dir;
    double win[QWN]; double dl[DLN]; double hs[SEQ_H], cs[SEQ_H];
} RamState;

static double ram_rate_map(double u)
{
    int i;
    if (u <= RAM_KNOTS[0]) return RAM_VALUES[0];
    if (u >= RAM_KNOTS[RAM_NK - 1]) return RAM_VALUES[RAM_NK - 1];
    for (i = 1; i < RAM_NK; i++)
        if (u <= RAM_KNOTS[i])
            return RAM_VALUES[i - 1] + (RAM_VALUES[i] - RAM_VALUES[i - 1]) * (u - RAM_KNOTS[i - 1]) / (RAM_KNOTS[i] - RAM_KNOTS[i - 1]);
    return RAM_VALUES[RAM_NK - 1];
}

static double ram_step(RamState *s, double z, double x0)
{
    int i; double lo, hi, mn, mx, u, rate, alpha = 1.0 - exp(-1.0 / (FS * OBS_TAU));
    int quiet = 0, is_quiet = 0; double delta, base;
    if (s->init == 0.0) {
        for (i = 0; i < QWN; i++) s->win[i] = z;
        for (i = 0; i < DLN; i++) s->dl[i] = 0.0;
        s->previous = z; s->estimate = OBS_CENTER; s->high_at = s->low_at = -1000000.0; s->x = x0; s->init = 1.0; s->k = 0.0;
    } else {
        for (i = QWN - 1; i > 0; i--) s->win[i] = s->win[i - 1];
        s->win[0] = z;
    }
    if (OBS_MODE != 0) {
        mn = mx = s->win[0];
        for (i = 1; i < QWN; i++) { if (s->win[i] < mn) mn = s->win[i]; if (s->win[i] > mx) mx = s->win[i]; }
        quiet = (mx - mn) <= OBS_QRANGE;
    }
    if (OBS_MODE != 0 && OBS_ADAPTIVE) {
        is_quiet = quiet && z >= OBS_GATE0 && z <= OBS_GATE1;
        if (is_quiet) {
            if (!s->was_quiet) { s->estimate = z; s->ready = 1.0; }
            else s->estimate += alpha * (z - s->estimate);
        } else if (OBS_MODE == 2) {
            double dir = (z - s->previous > 0) ? 1.0 : ((z - s->previous < 0) ? -1.0 : 0.0);
            double width = OBS_GATE1 - OBS_GATE0, mid, m2;
            if (s->prev_dir > 0 && dir < 0 && s->previous > OBS_GATE1 + width * 0.5) { s->high = s->previous; s->high_at = s->k; s->has_high = 1.0; }
            if (s->prev_dir < 0 && dir > 0 && s->previous < OBS_GATE0 - width * 0.5) { s->low = s->previous; s->low_at = s->k; s->has_low = 1.0; }
            m2 = s->high_at < s->low_at ? s->high_at : s->low_at;
            if (s->has_high != 0.0 && s->has_low != 0.0 && s->k - m2 < OBS_AGE) {
                mid = 0.5 * (s->high + s->low);
                if (mid >= OBS_GATE0 && mid <= OBS_GATE1) s->estimate += 0.05 * (mid - s->estimate);
            }
            if (dir != 0.0) s->prev_dir = dir;
        }
        s->was_quiet = is_quiet ? 1.0 : 0.0;
    } else {
        if (OBS_ADAPTIVE && z >= OBS_GATE0 && z <= OBS_GATE1 && fabs(z - s->previous) < OBS_MAXSTEP) {
            if (s->ready == 0.0) { s->estimate = z; s->ready = 1.0; }
            else s->estimate += alpha * (z - s->estimate);
        }
    }
    u = z - s->estimate;
    delta = z - s->previous;
    s->previous = z;
    if (RAM_DELAY > 0) {
        if (s->k == 0.0) for (i = 0; i < DLN; i++) s->dl[i] = u;
        { double out = s->dl[DLN - 1]; for (i = DLN - 1; i > 0; i--) s->dl[i] = s->dl[i - 1]; s->dl[0] = u; u = out; }
    }
    rate = ram_rate_map(u);
    if (RAM_HAS_ZB && u >= RAM_ZB0 && u <= RAM_ZB1) rate = 0.0;
    base = rate;
#if HAS_SEQ
    {
        double f[SEQ_NIN], pre[4 * SEQ_H], hn[SEQ_H], raw = SEQ_BIAS; int g, j;
        f[0] = z; f[1] = u; f[2] = s->estimate; f[3] = delta; f[4] = base;
        for (i = 0; i < SEQ_NIN; i++) f[i] = (f[i] - SEQ_MEAN[i]) / SEQ_SCALE[i];
        for (g = 0; g < 4 * SEQ_H; g++) {
            double a = SEQ_B[g];
            for (j = 0; j < SEQ_NIN; j++) a += SEQ_WI[g * SEQ_NIN + j] * f[j];
            for (j = 0; j < SEQ_H; j++) a += SEQ_WH[g * SEQ_H + j] * s->hs[j];
            pre[g] = a;
        }
        for (j = 0; j < SEQ_H; j++) {
            double ig = 1.0 / (1.0 + exp(-fmax(-60.0, fmin(60.0, pre[j]))));
            double fg = 1.0 / (1.0 + exp(-fmax(-60.0, fmin(60.0, pre[SEQ_H + j]))));
            double gg = tanh(pre[2 * SEQ_H + j]);
            double og = 1.0 / (1.0 + exp(-fmax(-60.0, fmin(60.0, pre[3 * SEQ_H + j]))));
            s->cs[j] = fg * s->cs[j] + ig * gg;
            hn[j] = og * tanh(s->cs[j]);
        }
        for (j = 0; j < SEQ_H; j++) { s->hs[j] = hn[j]; raw += hn[j] * SEQ_HEAD[j]; }
        rate = raw * SEQ_RSCALE;
        if (SEQ_RESIDUAL) rate += base;
        if (SEQ_GATE && base == 0.0) rate = 0.0;
        rate = fmin(SEQ_LIMIT, fmax(-SEQ_LIMIT, rate));
    }
#endif
    if (s->k > 0.0) {
        lo = RAM_LO; hi = RAM_HI;
        s->x = fmin(hi, fmax(lo, s->x + rate / FS));
    }
    s->k += 1.0;
    return s->x;
}

/* ---- Stage 3: RAM -> angle ---- */
static double angle_eval(double x)
{
    int i; double t = x / ANG_SCALE, y = ANG_W[ANG_N - 1];
    for (i = ANG_N - 2; i >= 0; i--) y = y * t + ANG_W[i];
    return y;
}
#endif
