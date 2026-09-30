"""Validated Link bytecode -> standalone dependency-free Rust."""
import shutil,subprocess
from pathlib import Path
from .syntax import LinkError
from .bridge import verify_links
HEADER=r'''#[derive(Clone,Copy)] struct Complex{re:f64,im:f64}
impl Complex{fn n(re:f64,im:f64)->Self{Self{re,im}} fn c(self)->Self{Self::n(self.re,-self.im)} fn a(self,b:Self)->Self{Self::n(self.re+b.re,self.im+b.im)} fn s(self,b:Self)->Self{Self::n(self.re-b.re,self.im-b.im)} fn m(self,b:Self)->Self{Self::n(self.re*b.re-self.im*b.im,self.re*b.im+self.im*b.re)}}
#[derive(Clone)] struct Matrix{rows:usize,cols:usize,d:Vec<Complex>}
impl Matrix{
fn new(r:usize,c:usize,d:Vec<Complex>)->Self{assert_eq!(d.len(),r*c);Self{rows:r,cols:c,d}}
fn identity(n:usize)->Self{let mut d=vec![Complex::n(0.,0.);n*n];for i in 0..n{d[i*n+i]=Complex::n(1.,0.)}Self::new(n,n,d)}
fn adjoint(&self)->Self{let mut d=vec![Complex::n(0.,0.);self.rows*self.cols];for i in 0..self.rows{for j in 0..self.cols{d[j*self.rows+i]=self.d[i*self.cols+j].c()}}Self::new(self.cols,self.rows,d)}
fn compose(a:&Self,b:&Self)->Self{assert_eq!(b.cols,a.rows);let mut d=vec![Complex::n(0.,0.);b.rows*a.cols];for i in 0..b.rows{for j in 0..a.cols{for k in 0..b.cols{let x=b.d[i*b.cols+k].m(a.d[k*a.cols+j]);let q=i*a.cols+j;d[q]=d[q].a(x)}}}Self::new(b.rows,a.cols,d)}
fn binary(a:&Self,b:&Self,sub:bool)->Self{assert_eq!((a.rows,a.cols),(b.rows,b.cols));Self::new(a.rows,a.cols,a.d.iter().zip(&b.d).map(|(x,y)|if sub{x.s(*y)}else{x.a(*y)}).collect())}
fn json(&self)->String{let mut o=format!("{{\"shape\":[{},{}],\"matrix\":[",self.rows,self.cols);for i in 0..self.rows{if i>0{o.push(',')}o.push('[');for j in 0..self.cols{if j>0{o.push(',')}let z=self.d[i*self.cols+j];o.push_str(&format!("{{\"re\":{},\"im\":{}}}",z.re,z.im))}o.push(']')}o.push_str("]}");o}}
'''
def _validate(p):
    if p.get('format')!='link-bytecode' or p.get('version')!=1:raise LinkError('Rust backend requires Link bytecode version 1')
    if p.get('orientation')!='input-rows-output-columns':raise LinkError('Rust backend requires input-rows-output-columns orientation')
    verify_links(p); shapes=[]
    for r in p['registers']:
        if r.get('dst')!=len(shapes):raise LinkError('Rust backend requires sequential register IDs')
        s=r['shape']; op=r['op']
        def ref(k):
            v=r.get(k)
            if type(v) is not int or not 0<=v<len(shapes):raise LinkError(f'Invalid register reference: {k}')
            return shapes[v]
        if op=='identity' and s[0]!=s[1]:raise LinkError('Identity must be square')
        elif op=='adjoint':
            a=ref('arg')
            if s!=[a[1],a[0]]:raise LinkError('Adjoint shape mismatch')
        elif op=='compose':
            a,b=ref('left'),ref('right')
            if a[0]!=b[1] or s!=[b[0],a[1]]:raise LinkError('Composition shape mismatch')
        elif op in ('add','sub'):
            a,b=ref('left'),ref('right')
            if a!=b or s!=a:raise LinkError('Addition/subtraction shape mismatch')
        elif op not in ('identity','matrix','phase2'):raise LinkError(f'Unsupported Rust opcode {op!r}')
        shapes.append(s)
def generate_rust(p):
    _validate(p); out=[HEADER,'fn main(){']
    for r in p['registers']:
        i=r['dst'];op=r['op'];s=r['shape']
        if op=='identity':e=f'Matrix::identity({s[0]})'
        elif op=='matrix':e=f"Matrix::new({s[0]},{s[1]},vec!["+",".join(f"Complex::n({float(z[0])!r},{float(z[1])!r})" for row in r['values'] for z in row)+"])"
        elif op=='phase2':
            z=[(1,0),(0,1),(-1,0),(0,-1)][r['q']];e=f'Matrix::new(1,1,vec![Complex::n({float(z[0])!r},{float(z[1])!r})])'
        elif op=='adjoint':e=f'r{r["arg"]}.adjoint()'
        elif op=='compose':e=f'Matrix::compose(&r{r["left"]},&r{r["right"]})'
        else:e=f'Matrix::binary(&r{r["left"]},&r{r["right"]},{str(op=="sub").lower()})'
        out.append(f'let r{i}={e};')
    out.append(f'print!("{{}}",r{p["result"]}.json());');out.append('}');return '\n'.join(out)+'\n'
def compile_rust(p,output,*,rustc='rustc',opt_level=2,source_path=None):
    output=Path(output); source=Path(source_path) if source_path else output.with_suffix('.rs'); source.write_text(generate_rust(p),encoding='utf-8')
    c=shutil.which(rustc)
    if c is None:raise RuntimeError(f'rustc not available: generated Rust source at {source}')
    r=subprocess.run([c,str(source),'-C',f'opt-level={opt_level}','-o',str(output)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stderr)
    return output
