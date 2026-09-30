"""Rust source backend compatibility tests."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from linkc.compiler import compile_source, optimize
from linkc.vm import run
from linkc.rust_backend import generate_rust, compile_rust

SOURCES = [
    'fn main() -> Link<C[2],C[2]> { return identity::<C[2]>(); }',
    '''fn main() -> Link<C[3],C[2]> {
       let $b = matrix::<C[3],C[4]>([[1,2,0,0],[0,1,0,0],[1,0,1,0]]);
       let $a = matrix::<C[4],C[2]>([[1,0],[0,1],[1,1],[0,0]]);
       return $a @ $b;
    }''',
    '''fn main() -> Link<C[2],C[1]> {
       let $x = matrix::<C[1],C[2]>([[i,-2]]);
       return $x†;
    }''',
    '''fn main() -> Link<C[2],C[2]> {
       let $a = identity::<C[2]>();
       let $b = matrix::<C[2],C[2]>([[1,2],[3,4]]);
       return $a + $b - $a;
    }''',
    'fn main() -> Link<C[1],C[1]> { return phase2(1.6) @ phase2(3.15); }',
]

class RustBackendTests(unittest.TestCase):
    def test_generate_structurally_valid_rust_for_all_ops(self):
        for src in SOURCES:
            with self.subTest(src=src):
                program = optimize(compile_source(src))
                rust = generate_rust(program)
                self.assertIn('fn main()', rust)
                self.assertIn('print!', rust)
                self.assertIn('struct Complex', rust)

    def test_rustc_missing_is_actionable(self):
        p = optimize(compile_source(SOURCES[0]))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'rustc'):
                compile_rust(p, Path(directory) / 'result', rustc='/nonexistent/link-rustc')

    @unittest.skipUnless(shutil.which('rustc'), 'requires installed rustc')
    def test_native_binary_matches_linkvm(self):
        for src in SOURCES:
            with self.subTest(src=src):
                p = optimize(compile_source(src))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'linked'
                    compile_rust(p, path)
                    actual = json.loads(subprocess.check_output([str(path)], text=True))
                    expected = run(p)
                    self.assertEqual(actual['shape'], expected['shape'])
                    for row1, row2 in zip(actual['matrix'], expected['matrix']):
                        for x, y in zip(row1, row2):
                            self.assertAlmostEqual(x['re'], y['re'], places=11)
                            self.assertAlmostEqual(x['im'], y['im'], places=11)

if __name__ == '__main__':
    unittest.main()

class BytecodeToRustCLITests(unittest.TestCase):
    def test_lbc_to_rust_cli(self):
        from linkc.cli import rust_main
        p = optimize(compile_source(SOURCES[1]))
        with tempfile.TemporaryDirectory() as d:
            src, generated = Path(d)/'p.lbc', Path(d)/'p.rs'
            src.write_text(json.dumps(p), encoding='utf-8')
            self.assertEqual(rust_main([str(src), '--emit-rust', str(generated)]),0)
            self.assertIn('Matrix::compose', generated.read_text())
            self.assertIn('matrix::<I,O> -> matrix -> matrix', generated.read_text())

class RustSafetyTests(unittest.TestCase):
    def test_bad_composition_bytecode_rejected(self):
        from linkc.syntax import LinkError
        p = optimize(compile_source(SOURCES[1]))
        p['registers'][2]['shape'] = [4,2]
        p.pop('operator_links')
        with self.assertRaisesRegex(LinkError,'Composition shape mismatch'):
            generate_rust(p)

    def test_corrupted_operator_certificate_rejected(self):
        from linkc.syntax import LinkError
        p = optimize(compile_source(SOURCES[1]))
        p['operator_links'][2]['bytecode'] = 'add'
        with self.assertRaisesRegex(LinkError,'Invalid operator Link translation'):
            generate_rust(p)
