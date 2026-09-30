use std::time::Instant;
const N:usize=100000;
type Mat=[f64;9]; type Vec3=[f64;3];
#[inline(always)] fn mul(a:&Mat,b:&Mat)->Mat { let mut c=[0.0;9]; for i in 0..3 {for j in 0..3{for k in 0..3{c[3*i+j]+=a[3*i+k]*b[3*k+j];}}} c }
#[inline(always)] fn apply(x:&Vec3,m:&Mat)->Vec3 {let mut y=[0.0;3];for j in 0..3{y[j]=x[0]*m[j]+x[1]*m[3+j]+x[2]*m[6+j];}y}
fn main(){
 let mut s:u32=20260930;
 let mut mats:Vec<Mat>=Vec::with_capacity(N);
 for i in 0..N {s=s.wrapping_mul(1664525).wrapping_add(1013904223);let q=((s>>24)&3)as f64;let theta=(q+1.0)*0.00013;let(c,v)=(theta.cos(),theta.sin());let mut a=[1.,0.,0.,0.,1.,0.,0.,0.,1.];match i%3{0=>{a[4]=c;a[8]=c;a[5]=-v;a[7]=v;},1=>{a[0]=c;a[8]=c;a[2]=v;a[6]=-v;},_=>{a[0]=c;a[4]=c;a[1]=-v;a[3]=v;}}mats.push(a);}
 for &width in &[1usize,10,100,1000,10000,100000] {
 let count=(N+width-1)/width;let t=Instant::now();let mut blocks=Vec::with_capacity(count);
 for chunk in mats.chunks(width){let mut acc:Mat=[1.,0.,0.,0.,1.,0.,0.,0.,1.];for m in chunk{acc=mul(&acc,m);}blocks.push(acc);}
 let build_ms=t.elapsed().as_secs_f64()*1000.0;
 let t=Instant::now();let mut sum=0.0;let queries=100usize;
 for qi in 0..queries{let mut x:Vec3=[0.1+qi as f64*0.001,0.4-qi as f64*0.0004,0.7+qi as f64*0.0001];for b in &blocks{x=apply(&x,b);}sum+=x[0];}
 std::hint::black_box(sum);let apply_ms=t.elapsed().as_secs_f64()*1000.0;
 let mut err=0.0_f64;
 for &qi in &[0usize,99]{let x:Vec3=[0.1+qi as f64*0.001,0.4-qi as f64*0.0004,0.7+qi as f64*0.0001];let mut refx=x;let mut out=x;for a in &mats{refx=apply(&refx,a);}for b in &blocks{out=apply(&out,b);}for j in 0..3{err=err.max((refx[j]-out[j]).abs());}}
 println!("width={width} blocks={count} queries=100 build_ms={build_ms:.6} apply_ms={apply_ms:.6} error={err:.12e} sum={sum:.9}");
 assert!(err<1e-10,"numerical discrepancy");
 }
}
