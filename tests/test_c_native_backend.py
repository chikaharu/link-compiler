import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from linkc.compiler import compile_source, optimize
from linkc.c_native_backend import compile_c, generate_c
from linkc.rust_backend import generate_rust
from linkc.vm import run

EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'

def example(name):
    return optimize(compile_source((EXAMPLES / f'{name}.link').read_text(encoding='utf-8')))

class NativeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('gcc'), 'gcc not installed')
    def test_native_execution_parity(self):
        for name in ('identity', 'rectangular', 'phases'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                p=example(name)
                executable=compile_c(p,Path(tmp)/name)
                actual=json.loads(subprocess.check_output([str(executable)]))
                expected=run(p)
                self.assertEqual(actual['shape'], expected['shape'])
                for row,row_exp in zip(actual['matrix'],expected['matrix']):
                    for cell,e in zip(row,row_exp):
                        self.assertAlmostEqual(cell['re'],e['re'],places=12)
                        self.assertAlmostEqual(cell['im'],e['im'],places=12)
    def test_emit_rust(self):
        p=example('rectangular')
        source=generate_rust(p)
        self.assertIn('fn main()',source)
        self.assertIn('Matrix::compose',source)
    def test_reject_invalid_shapes(self):
        p=example('identity')
        p['result_shape']=[4,3]
        with self.assertRaises(Exception):generate_c(p)
