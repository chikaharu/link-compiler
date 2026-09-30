"""Dependency-free LinkVM (register bytecode interpreter)."""
from .syntax import LinkError
from .bridge import verify_links


def _check_shape(matrix, shape):
    m, n = shape
    if not isinstance(m, int) or not isinstance(n, int) or m <= 0 or n <= 0:
        raise LinkError(f'Invalid positive matrix shape {shape!r}')
    if len(matrix) != m or any(len(row) != n for row in matrix):
        raise LinkError(f'Runtime shape mismatch: expected {m}x{n}')


def multiply(a, b):
    """Ordinary matrix product a * b; shape checks done by caller."""
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def _matrix_to_json(a):
    return [[{'re': z.real, 'im': z.imag} for z in row] for row in a]


def run(program):
    if program.get('format') != 'link-bytecode' or program.get('version') != 1:
        raise LinkError('Unsupported bytecode format or version')
    if program.get('orientation') != 'input-rows-output-columns':
        raise LinkError('Unsupported bytecode matrix orientation')
    verify_links(program)
    regs = {}
    element_states = {}
    for instruction in program.get('registers', []):
        op, idx, shape = instruction['op'], instruction['dst'], instruction['shape']
        if not isinstance(idx, int) or idx < 0 or idx in regs or idx != len(regs):
            raise LinkError('Bytecode register IDs must be sequential and unique')
        def ref(field):
            key = instruction[field]
            if not isinstance(key, int) or key not in regs:
                raise LinkError(f'Register %{idx} uses missing/future reference %{key}')
            return regs[key]
        m, n = shape
        if op == 'identity':
            if m != n:
                raise LinkError('Identity must be square')
            a = [[complex(i == j) for j in range(n)] for i in range(m)]
        elif op == 'matrix':
            vals = instruction['values']
            a = [[complex(*item) for item in row] for row in vals]
        elif op == 'element':
            if (m, n) != (1, 1):
                raise LinkError('ElementLink must have shape 1x1')
            value = complex(*instruction['value'])
            a = [[value]]
            element_states[idx] = {'value': value, 'pole_sign': 1, 'adjoint': False}
        elif op == 'element_neg':
            if (m, n) != (1, 1):
                raise LinkError('ElementLink negation must have shape 1x1')
            source_id = instruction['arg']
            a = [[ref('arg')[0][0]]]
            if source_id not in element_states:
                raise LinkError('ElementLink negation requires an ElementLink operand')
            state = dict(element_states[source_id])
            state['pole_sign'] *= -1
            element_states[idx] = state
        elif op == 'element_adjoint':
            if (m, n) != (1, 1):
                raise LinkError('ElementLink adjoint must have shape 1x1')
            source_id = instruction['arg']
            a = [[ref('arg')[0][0]]]
            if source_id not in element_states:
                raise LinkError('ElementLink adjoint requires an ElementLink operand')
            state = dict(element_states[source_id])
            state['adjoint'] = not state['adjoint']
            element_states[idx] = state
        elif op == 'element_link':
            if (m, n) != (1, 1):
                raise LinkError('ElementLink crosstalk must have shape 1x1')
            left_id, right_id = instruction['left'], instruction['right']
            ref('left')
            ref('right')
            if left_id not in element_states or right_id not in element_states:
                raise LinkError('ElementLink crosstalk requires ElementLink operands')

            def ports(state):
                value = state['value']
                sign = state['pole_sign']
                if state['adjoint']:
                    return 1j * value, sign * value
                return sign * value, 1j * value

            _, left_out = ports(element_states[left_id])
            right_in, _ = ports(element_states[right_id])
            value = left_out.conjugate() * right_in
            a = [[value]]
            element_states[idx] = {'value': value, 'pole_sign': 1, 'adjoint': False}
        elif op == 'phase2':
            if (m, n) != (1, 1) or instruction['q'] not in range(4):
                raise LinkError('Invalid phase2 bytecode')
            a = [[(1, 1j, -1, -1j)[instruction['q']]]]
        elif op == 'adjoint':
            x = ref('arg')
            a = [[x[j][i].conjugate() for j in range(len(x))] for i in range(len(x[0]))]
        elif op == 'compose':
            left, right = ref('left'), ref('right')
            if len(right[0]) != len(left):
                raise LinkError('Bytecode composition dimension mismatch')
            a = multiply(right, left)
        elif op in ('add', 'sub'):
            left, right = ref('left'), ref('right')
            if len(left) != len(right) or len(left[0]) != len(right[0]):
                raise LinkError('Bytecode addition/subtraction dimension mismatch')
            sign = 1 if op == 'add' else -1
            a = [[left[i][j] + sign * right[i][j] for j in range(len(left[0]))]
                 for i in range(len(left))]
        else:
            raise LinkError(f'Unknown bytecode opcode {op!r}')
        _check_shape(a, shape)
        regs[idx] = a
    result_id = program.get('result')
    if not isinstance(result_id, int) or result_id not in regs:
        raise LinkError('Invalid bytecode result register')
    answer = regs[result_id]
    _check_shape(answer, program['result_shape'])
    result = {'shape': program['result_shape'], 'matrix': _matrix_to_json(answer)}
    if result_id in element_states:
        state = element_states[result_id]
        result['element_state'] = {
            'pole_sign': state['pole_sign'],
            'adjoint': state['adjoint'],
        }
    return result
