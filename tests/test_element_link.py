"""Behavioral tests for directed ElementLink primitives."""
import unittest

from linkc.compiler import compile_source, optimize
from linkc.vm import run


class ElementLinkTests(unittest.TestCase):
    def test_link_of_element_links_uses_directed_crosstalk(self):
        src = """fn main() -> Link<C[1],C[1]> {
            let $a = element(1 + i);
            let $b = element(2 - i);
            return link($a, $b);
        }"""
        program = optimize(compile_source(src))
        self.assertEqual([r["op"] for r in program["registers"]],
                         ["element", "element", "element_link"])
        self.assertEqual(
            run(program)["matrix"][0][0],
            {"re": -3.0, "im": -1.0},
        )

    def test_element_adjoint_reverses_poles_and_is_involutive(self):
        src = """fn main() -> Link<C[1],C[1]> {
            let $a = element(1 + i);
            return $a†;
        }"""
        program = optimize(compile_source(src))
        self.assertEqual([r["op"] for r in program["registers"]],
                         ["element", "element_adjoint"])
        self.assertEqual(program["registers"][1]["arg"], 0)
        self.assertEqual(run(program)["matrix"][0][0],
                         {"re": 1.0, "im": 1.0})

        twice = """fn main() -> Link<C[1],C[1]> {
            let $a = element(1 + i);
            return $a††;
        }"""
        folded = optimize(compile_source(twice))
        self.assertEqual([r["op"] for r in folded["registers"]], ["element"])
        self.assertEqual(run(folded)["matrix"][0][0],
                         {"re": 1.0, "im": 1.0})


if __name__ == "__main__":
    unittest.main()
