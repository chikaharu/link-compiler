import unittest
from linkc.compiler import compile_source,optimize
from linkc.vm import run
from linkc.rust_backend import generate_rust
from linkc.syntax import LinkError

RECT = """fn main() -> Link<C[3], C[2]> {
 let $B: Link<C[3], C[4]> = matrix::<C[3], C[4]>([[1,2,0,0],[0,1,0,0],[1,0,1,0]]);
 let $A: Link<C[4], C[2]> = matrix::<C[4], C[2]>([[1,0],[0,1],[1,1],[0,0]]);
 return $A @ $B;
}"""

class SmokeTests(unittest.TestCase):
    def test_rectangular_link(self):
        p=optimize(compile_source(RECT))
        self.assertEqual(run(p)["shape"],[3,2])
        self.assertIn("Matrix::compose",generate_rust(p))
        self.assertEqual(p["operator_links"][-1]["source"],"@")

    def test_type_error(self):
        src="fn main() -> Link<C[2], C[2]> { let $A = identity::<C[2]>(); let $B = identity::<C[3]>(); return $A @ $B; }"
        with self.assertRaises(LinkError): compile_source(src)

if __name__=="__main__": unittest.main()
