package main
import("fmt";"math";"time")
const N=100000;const P=2000;const R=10;const Q=10
type Mat [9]float64
type Vec [3]float64
var sink uint64
func mul(a,b Mat)(c Mat){for i:=0;i<3;i++{for j:=0;j<3;j++{for k:=0;k<3;k++{c[i*3+j]+=a[i*3+k]*b[k*3+j]}}};return}
func apply(x Vec,a Mat)(y Vec){for j:=0;j<3;j++{y[j]=x[0]*a[j]+x[1]*a[3+j]+x[2]*a[6+j]};return}
func main(){q:=make([]uint8,N);mats:=make([]Mat,N);var s uint32=20260930;for i:=0;i<N;i++{s=s*1664525+1013904223;q[i]=uint8((s>>24)&3);th:=float64(q[i]+1)*.00013;c,v:=math.Cos(th),math.Sin(th);a:=Mat{1,0,0,0,1,0,0,0,1};switch i%3{case 0:a[4],a[8],a[5],a[7]=c,c,-v,v;case 1:a[0],a[8],a[2],a[6]=c,c,v,-v;default:a[0],a[4],a[1],a[3]=c,c,-v,v};mats[i]=a};expected:=0;for _,v:=range q{expected=(expected+int(v))&3};t:=time.Now();for r:=0;r<P;r++{x:=0;for _,v:=range q{x=(x+int(v))&3};sink+=uint64(x)};phase:=float64(time.Since(t).Nanoseconds())/1000/P;pre,seq,err:=0.,0.,0.;for rep:=0;rep<R;rep++{total:=Mat{1,0,0,0,1,0,0,0,1};t=time.Now();for _,m:=range mats{total=mul(total,m)};pre+=float64(time.Since(t).Nanoseconds())/1e6;t=time.Now();for query:=0;query<Q;query++{x:=Vec{.1+float64(query)*.001,.4-float64(query)*.0004,.7+float64(query)*.0001};input:=x;for _,m:=range mats{x=apply(x,m)};z:=apply(input,total);for j:=0;j<3;j++{err=math.Max(err,math.Abs(x[j]-z[j]))};sink+=uint64(math.Abs(x[0])*1e6)};seq+=float64(time.Since(t).Nanoseconds())/1e6};fmt.Printf("lang=Go phase_us=%.3f pre_ms=%.3f seq10_ms=%.3f result=%d error=%.3g sink=%d\n",phase,pre/R,seq/R,expected,err,sink);if expected!=3{panic("unexpected phase result")}}
