use std::time::Instant;
const N:usize=100000;const P:usize=2000;const R:usize=10;const Q:usize=10;
type M=[f64;9];type V=[f64;3];
#[inline]fn mul(a:&M,b:&M)->M{let mut c=[0.;9];for i in 0..3{for j in 0..3{for k in 0..3{c[i*3+j]+=a[i*3+k]*b[k*3+j];}}}c}
#[inline]fn apply(x:&V,a:&M)->V{let mut y=[0.;3];for j in 0..3{y[j]=x[0]*a[j]+x[1]*a[3+j]+x[2]*a[6+j];}y}
fn main(){
 let mut q=vec![0u8;N];let mut mats=vec![[0.;9];N];let mut s=20260930u32;
 for i in 0..N{
 s=s.wrapping_mul(1664525).wrapping_add(1013904223);q[i]=((s>>24)&3)as u8;
 let th=(q[i]as f64+1.)*.00013;let(c,v)=(th.cos(),th.sin());let mut a=[1.,0.,0.,0.,1.,0.,0.,0.,1.];
 match i%3{0=>{a[4]=c;a[8]=c;a[5]=-v;a[7]=v;},1=>{a[0]=c;a[8]=c;a[2]=v;a[6]=-v;},_=>{a[0]=c;a[4]=c;a[1]=-v;a[3]=v;}}mats[i]=a;
 }
 let expected=q.iter().fold(0usize,|x,&v|(x+v as usize)&3);
 let t=Instant::now();let mut sink:u64=0;
 for _ in 0..P{let mut x=0usize;for &v in &q{x=(x+v as usize)&3;}sink=sink.wrapping_add(x as u64);std::hint::black_box(sink);}
 let phase=t.elapsed().as_secs_f64()*1e6/P as f64;
 let(mut pre,mut seq,mut err)=(0.,0.,0.);
 for _ in 0..R{
 let mut total:M=[1.,0.,0.,0.,1.,0.,0.,0.,1.];let t=Instant::now();
 for m in &mats{total=mul(&total,m);}pre+=t.elapsed().as_secs_f64()*1e3;
 let t=Instant::now();
 for query in 0..Q{
 let mut x:V=[.1+query as f64*.001,.4-query as f64*.0004,.7+query as f64*.0001];let input=x;
 for m in &mats{x=apply(&x,m);}let z=apply(&input,&total);
 for j in 0..3{let d=(x[j]-z[j]).abs();if d>err{err=d;}}
 sink=sink.wrapping_add((x[0].abs()*1e6)as u64);std::hint::black_box(sink);
 }seq+=t.elapsed().as_secs_f64()*1e3;
 }
 println!("lang=Rust phase_us={:.3} pre_ms={:.3} seq10_ms={:.3} result={} error={:.3e} sink={}",phase,pre/R as f64,seq/R as f64,expected,err,sink);assert_eq!(expected,3);
}
