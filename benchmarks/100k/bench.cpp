#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
using Mat=std::array<double,9>;using Vec=std::array<double,3>;
constexpr int N=100000,P=2000,R=10,Q=10;
static volatile uint64_t sink=0;
static double ms(){using namespace std::chrono;return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();}
static Mat mul(const Mat&a,const Mat&b){Mat c{};for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)c[i*3+j]+=a[i*3+k]*b[k*3+j];return c;}
static Vec apply_link(const Vec&x,const Mat&a){Vec y{};for(int j=0;j<3;j++)y[j]=x[0]*a[j]+x[1]*a[3+j]+x[2]*a[6+j];return y;}
int main(){std::vector<uint8_t>q(N);std::vector<Mat>mats(N);uint32_t s=20260930u;for(int i=0;i<N;i++){s=s*1664525u+1013904223u;q[i]=(s>>24)&3;double th=(q[i]+1)*.00013,c=cos(th),v=sin(th);Mat a{1,0,0,0,1,0,0,0,1};if(i%3==0){a[4]=a[8]=c;a[5]=-v;a[7]=v;}else if(i%3==1){a[0]=a[8]=c;a[2]=v;a[6]=-v;}else{a[0]=a[4]=c;a[1]=-v;a[3]=v;}mats[i]=a;}
int expected=0;for(auto v:q)expected=(expected+v)&3;double t=ms();for(int r=0;r<P;r++){int x=0;for(auto v:q)x=(x+v)&3;sink=sink+(unsigned)x;}double phase=(ms()-t)*1000/P;double pre=0,seq=0,err=0;for(int rep=0;rep<R;rep++){Mat total{1,0,0,0,1,0,0,0,1};t=ms();for(const auto&m:mats)total=mul(total,m);pre+=ms()-t;t=ms();for(int query=0;query<Q;query++){Vec x{.1+query*.001,.4-query*.0004,.7+query*.0001};Vec input=x;for(const auto&m:mats)x=apply_link(x,m);Vec z=apply_link(input,total);for(int j=0;j<3;j++)err=std::max(err,std::abs(x[j]-z[j]));sink=sink+(uint64_t)(std::abs(x[0])*1000000);}seq+=ms()-t;}
std::cout<<std::fixed<<std::setprecision(3)<<"lang=C++ phase_us="<<phase<<" pre_ms="<<pre/R<<" seq10_ms="<<seq/R<<" result="<<expected<<" error="<<std::scientific<<err<<" sink="<<sink<<'\n';return expected!=3;}
