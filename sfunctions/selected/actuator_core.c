/* Generated coefficients, portable fixed-size states, no runtime JSON reader. */
#include "actuator_core.h"
#include "model_coefficients.h"
#include <math.h>
#include <string.h>
static double clamp(double x,double lo,double hi){return x<lo?lo:(x>hi?hi:x);}
static double sigmoid(double x){return 1.0/(1.0+exp(-clamp(x,-60.0,60.0)));}
size_t bb_ddv_state_size(void){return sizeof(BBDdvState);}
size_t bb_ram_state_size(void){return sizeof(BBRamState);}
void bb_ddv_init(BBDdvState *s){memset(s,0,sizeof(*s));}
void bb_ram_init(BBRamState *s,double x0){memset(s,0,sizeof(*s));s->x=x0;s->center=REF_CENTER;}
double bb_ddv_step(BBDdvState *s,double current){
    double z=current/DDV_SCALE,features[DDV_FEATURES],value=0.0;int i,j,p=0;
    if(!s->ready){for(i=0;i<DDV_HISTORY;i++)s->u[i]=z;s->ready=1;}
    else{for(i=DDV_HISTORY-1;i>0;i--)s->u[i]=s->u[i-1];s->u[0]=z;}
    features[p++]=1.0;
    for(i=0;i<DDV_HISTORY;i++)features[p++]=s->u[i];
#if DDV_RICH
    for(i=0;i<DDV_HISTORY;i++)features[p++]=fabs(s->u[i]);
    for(j=0;j<5;j++)for(i=0;i<DDV_HISTORY;i++)features[p++]=tanh(s->u[i]/DDV_TANH[j]);
    for(j=0;j<2;j++)for(i=0;i<DDV_HISTORY;i++)features[p++]=(s->u[i]>0?1:(s->u[i]<0?-1:0))*fmax(fabs(s->u[i])-DDV_DEAD[j],0.0);
#else
    (void)j;
    for(i=0;i<DDV_HISTORY;i++)features[p++]=tanh(s->u[i]);
    for(i=0;i<DDV_HISTORY;i++)features[p++]=tanh(s->u[i]/2.0);
    for(i=0;i<DDV_HISTORY;i++)features[p++]=fabs(s->u[i]);
#endif
    for(i=0;i<DDV_FEATURES;i++)value+=features[i]*DDV_WEIGHTS[i];
    return value;
}
static double lookup(double u){
    int lo=0,hi=REF_KNOTS-1,mid;
    if(u<=REF_U[0])return REF_RATE[0];
    if(u>=REF_U[hi])return REF_RATE[hi];
    while(hi-lo>1){mid=(lo+hi)/2;if(u<REF_U[mid])hi=mid;else lo=mid;}
    return REF_RATE[lo]+(u-REF_U[lo])/(REF_U[hi]-REF_U[lo])*(REF_RATE[hi]-REF_RATE[lo]);
}
double bb_ram_step(BBRamState *s,double ddv){
    double lo,hi,delta,u,base,feature[5],rate=0.0;
    int i,j,m,quiet;
    if(!s->ready){for(i=0;i<QUIET_WINDOW;i++)s->ddv[i]=ddv;s->previous=ddv;}
    else{for(i=QUIET_WINDOW-1;i>0;i--)s->ddv[i]=s->ddv[i-1];s->ddv[0]=ddv;}
    lo=hi=s->ddv[0];for(i=1;i<QUIET_WINDOW;i++){lo=fmin(lo,s->ddv[i]);hi=fmax(hi,s->ddv[i]);}
    quiet=hi-lo<=REF_QUIET_RANGE && ddv>=REF_GATE_LO && ddv<=REF_GATE_HI;
    if(REF_ADAPTIVE){if(quiet){if(!s->was_quiet)s->center=ddv;else s->center+=REF_ALPHA*(ddv-s->center);}s->was_quiet=quiet;}
    delta=ddv-s->previous;s->previous=ddv;u=ddv-s->center;
    if(!s->ready){for(i=0;i<=REF_DELAY;i++)s->delay[i]=u;}
    else{for(i=REF_DELAY;i>0;i--)s->delay[i]=s->delay[i-1];s->delay[0]=u;}
    u=s->delay[REF_DELAY];base=clamp(lookup(u),-REF_MAP_LIMIT,REF_MAP_LIMIT);
    if(REF_ZERO_BAND && u>=REF_ZERO_LO && u<=REF_ZERO_HI)base=0.0;
    feature[0]=ddv;feature[1]=u;feature[2]=s->center;feature[3]=delta;feature[4]=base;
    for(m=0;m<MEMBER_COUNT;m++){
        const BBWeights *w=&MEMBERS[m];double z[5],g[4*MAX_HIDDEN],rec[4*MAX_HIDDEN],raw=0.0,v;int h=w->hidden;
        for(i=0;i<5;i++)z[i]=(feature[i]-w->mean[i])/w->scale[i];
        for(i=0;i<w->gates*h;i++){
            g[i]=w->bi[i];rec[i]=w->bh[i];
            for(j=0;j<5;j++)g[i]+=w->wi[i*5+j]*z[j];
            for(j=0;j<h;j++)rec[i]+=w->wh[i*h+j]*s->hidden[m][j];
        }
        if(w->gates==4){
            for(i=0;i<h;i++){
                s->cell[m][i]=sigmoid(g[h+i]+rec[h+i])*s->cell[m][i]+sigmoid(g[i]+rec[i])*tanh(g[2*h+i]+rec[2*h+i]);
                s->hidden[m][i]=sigmoid(g[3*h+i]+rec[3*h+i])*tanh(s->cell[m][i]);
            }
        }else{
            for(i=0;i<h;i++){
                double reset=sigmoid(g[i]+rec[i]),update=sigmoid(g[h+i]+rec[h+i]);
                s->hidden[m][i]=(1.0-update)*tanh(g[2*h+i]+reset*rec[2*h+i])+update*s->hidden[m][i];
            }
        }
        raw=w->bias;for(i=0;i<h;i++)raw+=w->head[i]*s->hidden[m][i];
        v=clamp(raw*w->rate_scale,-w->correction_limit,w->correction_limit);
        if(w->residual)v+=base;
        if(w->gate && base==0.0)v=0.0;
        rate+=clamp(v,-w->output_limit,w->output_limit)/MEMBER_COUNT;
    }
    if(s->ready)s->x=clamp(s->x+rate/64.0,RAM_LO,RAM_HI);
    s->ready=1;return s->x;
}
double bb_angle(double ram){
    double z=ram/ANGLE_SCALE,value=ANGLE_COEFF[ANGLE_COUNT-1];int i;
    for(i=ANGLE_COUNT-2;i>=0;i--)value=value*z+ANGLE_COEFF[i];
    return value;
}
