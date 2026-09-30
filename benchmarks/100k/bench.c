#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#define N 100000
#define P 2000
#define R 10
#define Q 10
static uint8_t q[N];static double mats[N][9];static volatile uint64_t sink=0;
static double ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
static void mul(const double a[9],const double b[9],double c[9]){for(int i=0;i<3;i++)for(int j=0;j<3;j++){double v=0;for(int k=0;k<3;k++)v+=a[3*i+k]*b[3*k+j];c[3*i+j]=v;}}
static void apply(const double x[3],const double a[9],double y[3]){for(int j=0;j<3;j++)y[j]=x[0]*a[j]+x[1]*a[3+j]+x[2]*a[6+j];}
static void gen(){uint32_t s=20260930u;for(int i=0;i<N;i++){s=s*1664525u+1013904223u;q[i]=(s>>24)&3;double th=(q[i]+1)*.00013,c=cos(th),v=sin(th);double *a=mats[i];for(int k=0;k<9;k++)a[k]=0;a[0]=a[4]=a[8]=1;if(i%3==0){a[4]=a[8]=c;a[5]=-v;a[7]=v;}else if(i%3==1){a[0]=a[8]=c;a[2]=v;a[6]=-v;}else{a[0]=a[4]=c;a[1]=-v;a[3]=v;}}}
int main(){gen();int expected=0;for(int i=0;i<N;i++)expected=(expected+q[i])&3;double t=ms();for(int r=0;r<P;r++){int x=0;for(int i=0;i<N;i++)x=(x+q[i])&3;sink+=(unsigned)x;}double phase=(ms()-t)*1000/P;double total[9]={1,0,0,0,1,0,0,0,1},tmp[9],pre=0,seq=0,err=0;for(int rep=0;rep<R;rep++){for(int k=0;k<9;k++)total[k]=(k%4==0);t=ms();for(int i=0;i<N;i++){mul(total,mats[i],tmp);for(int k=0;k<9;k++)total[k]=tmp[k];}pre+=ms()-t;t=ms();for(int query=0;query<Q;query++){double x[3]={.1+query*.001,.4-query*.0004,.7+query*.0001},y[3],z[3],input[3]={x[0],x[1],x[2]};for(int i=0;i<N;i++){apply(x,mats[i],y);for(int j=0;j<3;j++)x[j]=y[j];}apply(input,total,z);for(int j=0;j<3;j++){double d=fabs(x[j]-z[j]);if(d>err)err=d;}sink+=(uint64_t)(fabs(x[0])*1000000);}seq+=ms()-t;}
printf("lang=C phase_us=%.3f pre_ms=%.3f seq10_ms=%.3f result=%d error=%.3g sink=%llu\n",phase,pre/R,seq/R,expected,err,(unsigned long long)sink);return expected!=3;}
