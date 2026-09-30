"""Typed sparse Links from surface-language operations to VM bytecode.

These are incidence maps on *operator symbols*, not on runtime numeric data.
An individual source operation e_s is carried through the semantic space into
one opcode e_b. Mathematical equivalence still requires typed compiler rules.
"""
from dataclasses import dataclass
from .syntax import LinkError

@dataclass(frozen=True)
class OperatorSpace:
    name: str
    basis: tuple[str, ...]

@dataclass(frozen=True)
class OperatorLink:
    source: OperatorSpace
    target: OperatorSpace
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if len({s for s, _ in self.edges}) != len(self.edges):
            raise ValueError('Each source operator needs exactly one destination')
        if any(s not in self.source.basis or t not in self.target.basis for s, t in self.edges):
            raise ValueError('Link edge outside operator spaces')

    def apply(self, symbol: str) -> str:
        for s, t in self.edges:
            if s == symbol:
                return t
        raise LinkError(f'Unlinked operator {symbol!r} in {self.source.name}')

    def sparse_matrix(self) -> list[list[int]]:
        """Rows = input symbols; columns = output symbols."""
        table = dict(self.edges)
        return [[int(table.get(s) == t) for t in self.target.basis]
                for s in self.source.basis]

    def then(self, other: 'OperatorLink') -> 'OperatorLink':
        if self.target != other.source:
            raise LinkError('Cannot compose operator Links with different intermediate types')
        return OperatorLink(self.source, other.target,
                            tuple((s, other.apply(t)) for s, t in self.edges))

SOURCE = OperatorSpace('LinkSourceOp', ('identity::<T>', 'matrix::<I,O>', 'phase2',
                                       '†', '@', '+', '-'))
SEMANTIC = OperatorSpace('TypedOperation', ('identity', 'matrix', 'phase2',
                                            'adjoint', 'compose', 'add', 'sub'))
BYTECODE = OperatorSpace('LinkVMOpcode', ('identity', 'matrix', 'phase2',
                                         'adjoint', 'compose', 'add', 'sub'))
SOURCE_TO_SEMANTIC = OperatorLink(SOURCE, SEMANTIC, (
    ('identity::<T>', 'identity'), ('matrix::<I,O>', 'matrix'),
    ('phase2', 'phase2'), ('†', 'adjoint'), ('@', 'compose'),
    ('+', 'add'), ('-', 'sub')))
SEMANTIC_TO_BYTECODE = OperatorLink(SEMANTIC, BYTECODE,
                                    tuple((op, op) for op in SEMANTIC.basis))
SOURCE_TO_BYTECODE = SOURCE_TO_SEMANTIC.then(SEMANTIC_TO_BYTECODE)

SOURCE_OF = {'identity': 'identity::<T>', 'matrix': 'matrix::<I,O>',
             'phase2': 'phase2', 'adjoint': '†', 'compose': '@',
             'add': '+', 'sub': '-'}


def translate(source_op: str) -> tuple[str, str]:
    semantic = SOURCE_TO_SEMANTIC.apply(source_op)
    opcode = SEMANTIC_TO_BYTECODE.apply(semantic)
    return semantic, opcode


def verify_links(program: dict) -> None:
    """Verify optional source->semantic->opcode certificates after lowering.

    A certificate is not executable authority; the VM continues to enforce
    opcode shapes, ordering and operand checks independently.
    """
    if 'operator_links' not in program:
        return
    links = program['operator_links']
    registers = program.get('registers', [])
    if not isinstance(links, list) or len(links) != len(registers):
        raise LinkError('Operator Link certificate must cover every live register')
    for i, (edge, reg) in enumerate(zip(links, registers)):
        if not isinstance(edge, dict) or edge.get('register') != i:
            raise LinkError(f'Invalid operator Link edge at %{i}')
        source_op = edge.get('source')
        semantic, opcode = translate(source_op)
        if edge.get('semantic') != semantic or edge.get('bytecode') != opcode:
            raise LinkError(f'Invalid operator Link translation at %{i}')
        if opcode != reg.get('op'):
            if reg.get('op') not in BYTECODE.basis:
                raise LinkError(f"Unknown bytecode opcode {reg.get('op')!r}")
            raise LinkError(f'Operator Link disagrees with bytecode at %{i}')
        if edge.get('shape') != reg.get('shape'):
            raise LinkError(f'Operator Link shape disagrees with bytecode at %{i}')


def dump_links(program: dict) -> str:
    verify_links(program)
    edges = program.get('operator_links', [])
    lines = ['LinkSourceOp -> TypedOperation -> LinkVMOpcode (live operator links)']
    for edge in edges:
        lines.append(f"%{edge['register']} Link<C[{edge['shape'][0]}],C[{edge['shape'][1]}]> "
                     f"{edge['source']} -> {edge['semantic']} -> {edge['bytecode']}")
    if not edges:
        lines.append('(no certificate; legacy bytecode)')
    return '\n'.join(lines)
