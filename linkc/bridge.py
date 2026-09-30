"""Typed sparse Links from surface-language operations to VM bytecode."""
from dataclasses import dataclass
from .syntax import LinkError
@dataclass(frozen=True)
class OperatorSpace: name:str; basis:tuple[str,...]
@dataclass(frozen=True)
class OperatorLink:
    source:OperatorSpace; target:OperatorSpace; edges:tuple[tuple[str,str],...]
    def __post_init__(self):
        if len({s for s,_ in self.edges})!=len(self.edges): raise ValueError('Each source operator needs exactly one destination')
        if any(s not in self.source.basis or t not in self.target.basis for s,t in self.edges): raise ValueError('Link edge outside operator spaces')
    def apply(self,symbol):
        for s,t in self.edges:
            if s==symbol:return t
        raise LinkError(f'Unlinked operator {symbol!r} in {self.source.name}')
    def sparse_matrix(self):
        table=dict(self.edges); return [[int(table.get(s)==t) for t in self.target.basis] for s in self.source.basis]
    def then(self,other):
        if self.target!=other.source: raise LinkError('Cannot compose operator Links with different intermediate types')
        return OperatorLink(self.source,other.target,tuple((s,other.apply(t)) for s,t in self.edges))
SOURCE=OperatorSpace('LinkSourceOp',('identity::<T>','matrix::<I,O>','phase2','†','@','+','-'))
SEMANTIC=OperatorSpace('TypedOperation',('identity','matrix','phase2','adjoint','compose','add','sub'))
BYTECODE=OperatorSpace('LinkVMOpcode',SEMANTIC.basis)
SOURCE_TO_SEMANTIC=OperatorLink(SOURCE,SEMANTIC,(('identity::<T>','identity'),('matrix::<I,O>','matrix'),('phase2','phase2'),('†','adjoint'),('@','compose'),('+','add'),('-','sub')))
SEMANTIC_TO_BYTECODE=OperatorLink(SEMANTIC,BYTECODE,tuple((x,x) for x in SEMANTIC.basis))
SOURCE_TO_BYTECODE=SOURCE_TO_SEMANTIC.then(SEMANTIC_TO_BYTECODE)
SOURCE_OF={'identity':'identity::<T>','matrix':'matrix::<I,O>','phase2':'phase2','adjoint':'†','compose':'@','add':'+','sub':'-'}
def translate(source_op):
    semantic=SOURCE_TO_SEMANTIC.apply(source_op); return semantic,SEMANTIC_TO_BYTECODE.apply(semantic)
def verify_links(program):
    if 'operator_links' not in program:return
    links=program['operator_links']; regs=program.get('registers',[])
    if not isinstance(links,list) or len(links)!=len(regs): raise LinkError('Operator Link certificate must cover every live register')
    for i,(edge,reg) in enumerate(zip(links,regs)):
        if not isinstance(edge,dict) or edge.get('register')!=i: raise LinkError(f'Invalid operator Link edge at %{i}')
        semantic,opcode=translate(edge.get('source'))
        if edge.get('semantic')!=semantic or edge.get('bytecode')!=opcode: raise LinkError(f'Invalid operator Link translation at %{i}')
        if opcode!=reg.get('op'): raise LinkError(f'Operator Link disagrees with bytecode at %{i}')
        if edge.get('shape')!=reg.get('shape'): raise LinkError(f'Operator Link shape disagrees with bytecode at %{i}')
def dump_links(program):
    verify_links(program); lines=['LinkSourceOp -> TypedOperation -> LinkVMOpcode (live operator links)']
    for e in program.get('operator_links',[]): lines.append(f"%{e['register']} Link<C[{e['shape'][0]}],C[{e['shape'][1]}]> {e['source']} -> {e['semantic']} -> {e['bytecode']}")
    if len(lines)==1:lines.append('(no certificate; legacy bytecode)')
    return '\n'.join(lines)
