"""Typed lowering and conservative optimizations for Link register IR."""
import math
from .syntax import parse, LinkError
from .bridge import SOURCE_OF, translate, verify_links


def _complex_json(x):
    x = complex(x)
    return [x.real, x.imag]


def _scalar(node):
    op = node[0]
    if op == 'num':
        return node[1]
    if op == 'neg':
        return -_scalar(node[1])
    if op in ('add', 'sub'):
        x, y = _scalar(node[1]), _scalar(node[2])
        return x + y if op == 'add' else x - y
    raise LinkError('Matrix entries and phase2 angles must be compile-time scalar constants')


class Compiler:
    def __init__(self):
        self.program = []
        self.types = {}
        self.vars = {}
        self.identity = set()
        self.q2 = {}
        self.operator_links = []

    def emit(self, op, shape, **fields):
        reg = len(self.program)
        source = SOURCE_OF[op]
        semantic, opcode = translate(source)
        self.program.append(dict(op=opcode, dst=reg, shape=list(shape), **fields))
        self.operator_links.append(dict(register=reg, source=source, semantic=semantic,
                                        bytecode=opcode, shape=list(shape)))
        self.types[reg] = shape
        return reg

    def expr(self, node):
        op = node[0]
        if op == 'var':
            name = node[1]
            if name not in self.vars:
                raise LinkError(f'Undefined variable {name}')
            return self.vars[name]
        if op == 'identity':
            shape = (node[1], node[2])
            reg = self.emit('identity', shape)
            self.identity.add(reg)
            return reg
        if op == 'matrix':
            rows, cols, raw = node[1:]
            if len(raw) != rows or any(not isinstance(row, list) or len(row) != cols for row in raw):
                raise LinkError(f'Matrix literal requires {rows} input rows and {cols} output columns')
            vals = [[_complex_json(_scalar(value)) for value in row] for row in raw]
            return self.emit('matrix', (rows, cols), values=vals)
        if op == 'phase2':
            theta = _scalar(node[1])
            if complex(theta).imag:
                raise LinkError('phase2 angle must be real')
            # Nearest quarter-turn, ties defined by floor(x+0.5).
            q = math.floor((float(theta) / (math.pi / 2)) + 0.5) % 4
            reg = self.emit('phase2', (1, 1), q=q)
            self.q2[reg] = q
            return reg
        if op == 'adjoint':
            source = self.expr(node[1])
            m, n = self.types[source]
            if source in self.identity:
                return source
            if source in self.q2:
                q = (-self.q2[source]) % 4
                reg = self.emit('phase2', (1, 1), q=q)
                self.q2[reg] = q
                return reg
            return self.emit('adjoint', (n, m), arg=source)
        if op in ('compose', 'add', 'sub'):
            left, right = self.expr(node[1]), self.expr(node[2])
            a, b = self.types[left], self.types[right]
            if op == 'compose':
                # A @ B = A after B: B: X->Y, A: Y->Z.
                if a[0] != b[1]:
                    raise LinkError(f'Cannot compose {a} @ {b}: left input dimension {a[0]} != right output dimension {b[1]}')
                shape = (b[0], a[1])
                if left in self.identity:
                    return right
                if right in self.identity:
                    return left
                if left in self.q2 and right in self.q2:
                    q = (self.q2[left] + self.q2[right]) % 4
                    reg = self.emit('phase2', (1, 1), q=q)
                    self.q2[reg] = q
                    return reg
                return self.emit('compose', shape, left=left, right=right)
            if a != b:
                raise LinkError(f'Cannot {op} Links with shapes {a} and {b}')
            return self.emit(op, a, left=left, right=right)
        raise LinkError(f'Expression not supported as a Link: {op}')

    def compile(self, src):
        result_shape, stmts = parse(src)
        for stmt in stmts:
            if stmt[0] == 'let':
                _, name, declared, expression = stmt
                if name in self.vars:
                    raise LinkError(f'Variable {name} is already defined (use a new name)')
                reg = self.expr(expression)
                actual = self.types[reg]
                if declared is not None and actual != declared:
                    raise LinkError(f'Variable {name} declared Link<C{declared}> but inferred Link<C{actual}>')
                self.vars[name] = reg
            else:
                reg = self.expr(stmt[1])
                if self.types[reg] != result_shape:
                    raise LinkError(f'Main declares Link<C{result_shape}> but returns Link<C{self.types[reg]}>')
                return dict(format='link-bytecode', version=1, orientation='input-rows-output-columns',
                            registers=self.program, result=reg, result_shape=list(result_shape),
                            operator_links=self.operator_links)
        raise LinkError('Missing return')


def compile_source(src):
    return Compiler().compile(src)


def optimize(program):
    """Remove dead registers and remap references; compile already folds identities/q2."""
    registers = program['registers']
    by_id = {r['dst']: r for r in registers}
    live = set()

    def visit(idx):
        if idx in live:
            return
        if idx not in by_id:
            raise LinkError(f'Unknown LIR register %{idx}')
        live.add(idx)
        r = by_id[idx]
        for key in ('arg', 'left', 'right'):
            if key in r:
                visit(r[key])

    visit(program['result'])
    ordered = [r for r in registers if r['dst'] in live]
    mapping = {r['dst']: i for i, r in enumerate(ordered)}
    compact = []
    for i, old in enumerate(ordered):
        r = dict(old)
        r['dst'] = i
        for key in ('arg', 'left', 'right'):
            if key in r:
                r[key] = mapping[r[key]]
        compact.append(r)
    result = {**program, 'registers': compact, 'result': mapping[program['result']]}
    if 'operator_links' in program:
        result['operator_links'] = [dict(program['operator_links'][old['dst']], register=i)
                                    for i, old in enumerate(ordered)]
        verify_links(result)
    return result


def lir(program):
    lines = []
    for r in program['registers']:
        shape = r['shape']
        details = {k: v for k, v in r.items() if k not in ('dst', 'shape', 'op')}
        lines.append(f"%{r['dst']}: Link<C[{shape[0]}],C[{shape[1]}]> = {r['op']} {details}")
    lines.append(f"return %{program['result']}")
    return '\n'.join(lines)
