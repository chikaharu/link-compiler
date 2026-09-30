"""Numerical and structural tests for typed language-to-bytecode Links."""
import copy
import unittest
from linkc.bridge import (OperatorLink, OperatorSpace, SOURCE, SEMANTIC, BYTECODE,
                          SOURCE_TO_SEMANTIC, SEMANTIC_TO_BYTECODE,
                          SOURCE_TO_BYTECODE, verify_links, dump_links)
from linkc.compiler import compile_source, optimize
from linkc.syntax import LinkError
from linkc.vm import run


class OperatorLinkTests(unittest.TestCase):
    def test_sparse_matrix_matches_translation(self):
        a = SOURCE_TO_SEMANTIC.sparse_matrix()
        b = SEMANTIC_TO_BYTECODE.sparse_matrix()
        c = SOURCE_TO_BYTECODE.sparse_matrix()
        self.assertEqual([len(row) for row in a], [len(SEMANTIC.basis)] * len(SOURCE.basis))
        self.assertTrue(all(sum(row) == 1 for row in a))
        self.assertEqual(c, [[sum(a[i][k] * b[k][j] for k in range(len(SEMANTIC.basis)))
                              for j in range(len(BYTECODE.basis))] for i in range(len(SOURCE.basis))])
        self.assertEqual(SOURCE_TO_BYTECODE.apply('@'), 'compose')
        self.assertEqual(SOURCE_TO_BYTECODE.apply('†'), 'adjoint')

    def test_type_guard(self):
        wrong = OperatorSpace('Different', ('adjoint',))
        bad = OperatorLink(wrong, BYTECODE, (('adjoint', 'adjoint'),))
        with self.assertRaisesRegex(LinkError, 'intermediate types'):
            SOURCE_TO_SEMANTIC.then(bad)
        with self.assertRaisesRegex(LinkError, 'Unlinked operator'):
            SOURCE_TO_SEMANTIC.apply('unlisted')

    def test_compiled_links_and_numerical_semantics(self):
        src = '''fn main() -> Link<C[3],C[2]> {
             let $B = matrix::<C[3],C[4]>([[1,2,0,0],[0,1,0,0],[1,0,1,0]]);
             let $A = matrix::<C[4],C[2]>([[1,0],[0,1],[1,1],[0,0]]);
             return $A @ $B;
        }'''
        program = optimize(compile_source(src))
        verify_links(program)
        self.assertIn('@ -> compose -> compose', dump_links(program))
        self.assertEqual(run(program)['matrix'][2], [{'re': 2.0, 'im': 0.0}, {'re': 1.0, 'im': 0.0}])
        self.assertEqual([e['source'] for e in program['operator_links']],
                         ['matrix::<I,O>', 'matrix::<I,O>', '@'])

    def test_dead_code_remaps_certificate(self):
        src = '''fn main() -> Link<C[2],C[2]> {
             let $unused = matrix::<C[1],C[1]>([[7]]);
             let $live = identity::<C[2]>();
             return $live;
        }'''
        program = optimize(compile_source(src))
        self.assertEqual(len(program['operator_links']), 1)
        self.assertEqual(program['operator_links'][0]['register'], 0)
        self.assertEqual(run(program)['shape'], [2, 2])

    def test_tampering_detected(self):
        program = optimize(compile_source('fn main() -> Link<C[1],C[1]> { return identity::<C[1]>(); }'))
        bad = copy.deepcopy(program)
        bad['operator_links'][0]['source'] = '@'
        with self.assertRaisesRegex(LinkError, 'translation'):
            run(bad)
        bad = copy.deepcopy(program)
        bad['operator_links'][0]['shape'] = [7, 8]
        with self.assertRaisesRegex(LinkError, 'shape'):
            run(bad)

    def test_legacy_bytecode_is_still_supported(self):
        program = compile_source('fn main() -> Link<C[1],C[1]> { return phase2(1.6); }')
        del program['operator_links']
        self.assertEqual(run(program)['matrix'][0][0], {'re': 0.0, 'im': 1.0})


if __name__ == '__main__':
    unittest.main()
