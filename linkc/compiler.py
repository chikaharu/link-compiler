"""Typed lowering and conservative optimizations for Link register IR."""
import math
from .syntax import parse,LinkError
from .bridge import SOURCE_OF,translate,verify_links
def _complex_json(x): x=complex(x); return [x.real,x.imag]
def _scalar(n):
    if n[0]=='num': return n[1]
    if n[0]=='neg': return -_scalar(n[1])
    if n[0] in ('add','sub'):
        x,y=_scalar(n[1]),_scalar(n[2]); return x+y if n[0]=='add' else x-y
    raise LinkError('Matrix entries and phase2 angles must be compile-time scalar constants')
class Compiler:
    def __init__(self): self.program=[]; self.types={}; self.vars={}; self.identity=set(); self.q2={}; self.operator_links=[]
    def emit(self,op,shape,**fields):
        reg=len(self.program); source=SOURCE_OF[op]; semantic,opcode=translate(source)
        self.program.append(dict(op=opcode,dst=reg,shape=list(shape),**fields))
        self.operator_links.append(dict(register=reg,source=source,semantic=semantic,bytecode=opcode,shape=list(shape))); self.types[reg]=shape; return reg
    def expr(self,n):
        op=n[0]
        if op=='var':
            if n[1] not in self.vars: raise LinkError(f'Undefined variable {n[1]}')
            return self.vars[n[1]]
        if op=='identity':
            r=self.emit('identity',(n[1],n[2])); self.identity.add(r); return r
        if op=='matrix':
            rows,cols,raw=n[1:]
            if len(raw)!=rows or any(not isinstance(row,list) or len(row)!=cols for row in raw): raise LinkError(f'Matrix literal requires {rows} input rows and {cols} output columns')
            return self.emit('matrix',(rows,cols),values=[[_complex_json(_scalar(v)) for v in row] for row in raw])
        if op=='phase2':
            theta=_scalar(n[1])
            if complex(theta).imag: raise LinkError('phase2 angle must be real')
            q=math.floor(float(theta)/(math.pi/2)+.5)%4; r=self.emit('phase2',(1,1),q=q); self.q2[r]=q; return r
        if op=='adjoint':
            s=self.expr(n[1]); m,k=self.types[s]
            if s in self.identity: return s
            if s in self.q2:
                q=(-self.q2[s])%4; r=self.emit('phase2',(1,1),q=q); self.q2[r]=q; return r
            return self.emit('adjoint',(k,m),arg=s)
        if op in ('compose','add','sub'):
            l,r=self.expr(n[1]),self.expr(n[2]); a,b=self.types[l],self.types[r]
            if op=='compose':
                if a[0]!=b[1]: raise LinkError(f'Cannot compose {a} @ {b}: left input dimension {a[0]} != right output dimension {b[1]}')
                shape=(b[0],a[1])
                if l in self.identity:return r
                if r in self.identity:return l
                if l in self.q2 and r in self.q2:
                    q=(self.q2[l]+self.q2[r])%4; x=self.emit('phase2',(1,1),q=q); self.q2[x]=q; return x
                return self.emit('compose',shape,left=l,right=r)
            if a!=b: raise LinkError(f'Cannot {op} Links with shapes {a} and {b}')
            return self.emit(op,a,left=l,right=r)
        raise LinkError(f'Expression not supported as a Link: {op}')
    def compile(self,src):
        result_shape,stmts=parse(src)
        for s in stmts:
            if s[0]=='let':
                _,name,decl,expr=s
                if name in self.vars: raise LinkError(f'Variable {name} is already defined (use a new name)')
                r=self.expr(expr); actual=self.types[r]
                if decl is not None and actual!=decl: raise LinkError(f'Variable {name} declared Link<C{decl}> but inferred Link<C{actual}>')
                self.vars[name]=r
            else:
                r=self.expr(s[1])
                if self.types[r]!=result_shape: raise LinkError(f'Main declares Link<C{result_shape}> but returns Link<C{self.types[r]}>')
                return dict(format='link-bytecode',version=1,orientation='input-rows-output-columns',registers=self.program,result=r,result_shape=list(result_shape),operator_links=self.operator_links)
        raise LinkError('Missing return')
def compile_source(src): return Compiler().compile(src)
def optimize(program):
    regs=program['registers']; by={r['dst']:r for r in regs}; live=set()
    def visit(i):
        if i in live:return
        if i not in by: raise LinkError(f'Unknown LIR register %{i}')
        live.add(i)
        for k in ('arg','left','right'):
            if k in by[i]: visit(by[i][k])
    visit(program['result']); ordered=[r for r in regs if r['dst'] in live]; mapping={r['dst']:i for i,r in enumerate(ordered)}; compact=[]
    for i,old in enumerate(ordered):
        r=dict(old); r['dst']=i
        for k in ('arg','left','right'):
            if k in r:r[k]=mapping[r[k]]
        compact.append(r)
    out={**program,'registers':compact,'result':mapping[program['result']]}
    if 'operator_links' in program:
        out['operator_links']=[dict(program['operator_links'][old['dst']],register=i) for i,old in enumerate(ordered)]; verify_links(out)
    return out
def lir(program):
    lines=[]
    for r in program['registers']:
        s=r['shape']; d={k:v for k,v in r.items() if k not in ('dst','shape','op')}; lines.append(f"%{r['dst']}: Link<C[{s[0]}],C[{s[1]}]> = {r['op']} {d}")
    lines.append(f"return %{program['result']}"); return '\n'.join(lines)
