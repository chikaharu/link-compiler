import json
import unittest
from linkc.syntax import LinkError, parse
from linkc.compiler import compile_source, optimize
from linkc.vm import run


class LinkCompilerTests(unittest.TestCase):
    def compile_and_run(self, source):
        return run(optimize(compile_source(source)))

    def test_identity_optimization(self):
        source = 'fn main() -> Link<C[4],C[4]> { let $L = identity::<C[4]>(); return $L† @ $L; }'
        program = optimize(compile_source(source))
        self.assertEqual([i['op'] for i in program['registers']], ['identity'])
        result = run(program)['matrix']
        self.assertEqual([[z['re'] for z in row] for row in result],
                         [[float(i == j) for j in range(4)] for i in range(4)])

    def test_rectangular_compose_orientation(self):
        src = '''fn main() -> Link<C[3],C[2]> {
            let $B = matrix::<C[3],C[4]>([[1,2,0,0],[0,1,0,0],[1,0,1,0]]);
            let $A = matrix::<C[4],C[2]>([[1,0],[0,1],[1,1],[0,0]]);
            return $A @ $B;
        }'''
        out = self.compile_and_run(src)
        self.assertEqual(out['shape'], [3,2])
        self.assertEqual([[v['re'] for v in row] for row in out['matrix']],
                         [[1,2],[0,1],[2,1]])

    def test_adjoint_rectangular(self):
        src = 'fn main() -> Link<C[2],C[1]> { let $A = matrix::<C[1],C[2]>([[1+i, -2]]); return $A†; }'
        # Scalars deliberately disallow complex expressions in initial grammar; use i alone.
        src = 'fn main() -> Link<C[2],C[1]> { let $A = matrix::<C[1],C[2]>([[i, -2]]); return $A†; }'
        out = self.compile_and_run(src)
        self.assertEqual(out['matrix'], [[{'re': 0.0, 'im': -1.0}], [{'re': -2.0, 'im': 0.0}]])

    def test_phase2_modular_folding(self):
        src = 'fn main() -> Link<C[1],C[1]> { let $a = phase2(1.6); let $b = phase2(3.15); return $a @ $b; }'
        program = optimize(compile_source(src))
        self.assertEqual([x['op'] for x in program['registers']], ['phase2'])
        self.assertEqual(program['registers'][0]['q'], 3)
        self.assertEqual(run(program)['matrix'][0][0], {'re': 0.0, 'im': -1.0})

    def test_type_mismatch(self):
        src = 'fn main() -> Link<C[2],C[3]> { let $a = identity::<C[2]>(); return $a; }'
        with self.assertRaisesRegex(LinkError, 'declares'):
            compile_source(src)

    def test_bad_composition(self):
        src = '''fn main() -> Link<C[2],C[2]> {
          let $a = matrix::<C[2],C[3]>([[1,0,0],[0,1,0]]);
          let $b = identity::<C[2]>();
          return $a @ $b;
        }'''
        with self.assertRaisesRegex(LinkError, 'declares'):
            compile_source(src)
        src = '''fn main() -> Link<C[2],C[2]> {
          let $a = matrix::<C[3],C[2]>([[1,0],[0,1],[0,0]]);
          let $b = identity::<C[4]>();
          return $a @ $b;
        }'''
        with self.assertRaisesRegex(LinkError, 'Cannot compose'):
            compile_source(src)

    def test_matrix_shape_validation(self):
        src = 'fn main() -> Link<C[2],C[2]> { return matrix::<C[2],C[2]>([[1,2]]); }'
        with self.assertRaisesRegex(LinkError, 'requires 2 input rows'):
            compile_source(src)

    def test_unknown_char(self):
        with self.assertRaisesRegex(LinkError, 'Unexpected character'):
            parse('fn main() -> Link<C[1],C[1]> { return !; }')

    def test_bad_vm_opcode_and_version(self):
        data = compile_source('fn main() -> Link<C[1],C[1]> { return identity::<C[1]>(); }')
        data['registers'][0]['op'] = 'EXEC_PYTHON'
        with self.assertRaisesRegex(LinkError, 'Unknown bytecode'):
            run(data)
        data['version'] = 987
        with self.assertRaisesRegex(LinkError, 'Unsupported bytecode'):
            run(data)

    def test_dead_code_elimination(self):
        src = '''fn main() -> Link<C[2],C[2]> {
            let $unused = matrix::<C[1],C[1]>([[7]]);
            return identity::<C[2]>();
        }'''
        original = compile_source(src)
        self.assertEqual(len(original['registers']), 2)
        optimized = optimize(original)
        self.assertEqual(len(optimized['registers']), 1)

    def test_add_sub(self):
        src = '''fn main() -> Link<C[2],C[2]> {
            let $a = identity::<C[2]>();
            let $b = matrix::<C[2],C[2]>([[1,2],[3,4]]);
            return $a + $b - $a;
        }'''
        out = self.compile_and_run(src)
        self.assertEqual([[z['re'] for z in row] for row in out['matrix']], [[1,2],[3,4]])

if __name__ == '__main__':
    unittest.main()
