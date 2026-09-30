"""Fallback native backend: verified Link bytecode -> standalone C -> GCC executable.

This is explicitly NOT a Rust binary; use rust_backend.compile_rust when rustc is available.
"""
import shutil
import subprocess
from pathlib import Path
from .rust_backend import _validate, _number

C_HEADER = r'''/* linkc native C fallback: input rows, output columns. */
#include <stdio.h>
#include <stdlib.h>
typedef struct { double re,im; } Z;
typedef struct { size_t rows,cols; Z *d; } M;
static Z add(Z a,Z b){return (Z){a.re+b.re,a.im+b.im};}
static Z sub(Z a,Z b){return (Z){a.re-b.re,a.im-b.im};}
static Z mul(Z a,Z b){return (Z){a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};}
static M matrix(size_t m,size_t n){M x={m,n,calloc(m*n,sizeof(Z))};if(!x.d)exit(3);return x;}
static M identity(size_t n){M x=matrix(n,n);for(size_t i=0;i<n;i++)x.d[i*n+i].re=1.;return x;}
static M adjoint(M a){M x=matrix(a.cols,a.rows);for(size_t i=0;i<a.rows;i++)for(size_t j=0;j<a.cols;j++){Z z=a.d[i*a.cols+j];x.d[j*a.rows+i]=(Z){z.re,-z.im};}return x;}
/* A @ B applies B then A; stored product B*A. */
static M compose(M a,M b){if(b.cols!=a.rows)exit(4);M x=matrix(b.rows,a.cols);for(size_t i=0;i<b.rows;i++)for(size_t j=0;j<a.cols;j++)for(size_t k=0;k<b.cols;k++)x.d[i*a.cols+j]=add(x.d[i*a.cols+j],mul(b.d[i*b.cols+k],a.d[k*a.cols+j]));return x;}
static M binary(M a,M b,int subtract){if(a.rows!=b.rows||a.cols!=b.cols)exit(4);M x=matrix(a.rows,a.cols);for(size_t i=0;i<a.rows*a.cols;i++)x.d[i]=subtract?sub(a.d[i],b.d[i]):add(a.d[i],b.d[i]);return x;}
static void emit(M a){printf("{\"shape\":[%zu,%zu],\"matrix\":[",a.rows,a.cols);for(size_t i=0;i<a.rows;i++){if(i)putchar(',');putchar('[');for(size_t j=0;j<a.cols;j++){if(j)putchar(',');Z z=a.d[i*a.cols+j];printf("{\"re\":%.17g,\"im\":%.17g}",z.re,z.im);}putchar(']');}printf("]}");}
'''

def generate_c(program):
    _validate(program)
    out = [C_HEADER, 'int main(void) {']
    for r in program['registers']:
        op,i=r['op'],r['dst']; m,n=r['shape']
        if op=='identity': out.append(f' M r{i}=identity({n});')
        elif op=='matrix':
            out.append(f' M r{i}=matrix({m},{n});')
            for row_i,row in enumerate(r['values']):
                for col_i,(re,im) in enumerate(row):
                    out.append(f' r{i}.d[{row_i*n+col_i}]=(Z){{{_number(re)},{_number(im)}}};')
        elif op=='phase2':
            re,im=[(1,0),(0,1),(-1,0),(0,-1)][r['q']]
            out.append(f' M r{i}=matrix(1,1);r{i}.d[0]=(Z){{{re}.0,{im}.0}};')
        elif op=='adjoint': out.append(f' M r{i}=adjoint(r{r["arg"]});')
        elif op=='compose': out.append(f' M r{i}=compose(r{r["left"]},r{r["right"]});')
        elif op in ('add','sub'): out.append(f' M r{i}=binary(r{r["left"]},r{r["right"]},{int(op=="sub")});')
    out.append(f' emit(r{program["result"]});')
    out.append(' return 0;\n}')
    return '\n'.join(out)+'\n'

def compile_c(program, output, *, cc='gcc', source_path=None):
    target=Path(output)
    source=Path(source_path) if source_path else target.with_suffix('.c')
    source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text(generate_c(program),encoding='utf-8')
    compiler=shutil.which(cc)
    if compiler is None: raise RuntimeError(f'Native C compiler not available: {cc}')
    proc=subprocess.run([compiler,'-std=c11','-O2','-Wall','-Werror','-Wno-unused-function',str(source),'-o',str(target)],capture_output=True,text=True)
    if proc.returncode: raise RuntimeError(proc.stderr)
    return target
