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


    def test_element_negation_flips_only_the_element_input_pole(self):
        left_neg = """fn main() -> Link<C[1],C[1]> {
            let $a = element(1 + i);
            let $b = element(2 - i);
            return link(-$a, $b);
        }"""
        right_neg = """fn main() -> Link<C[1],C[1]> {
            let $a = element(1 + i);
            let $b = element(2 - i);
            return link($a, -$b);
        }"""

        left = optimize(compile_source(left_neg))
        right = optimize(compile_source(right_neg))

        self.assertEqual([r["op"] for r in left["registers"]],
                         ["element", "element", "element_neg", "element_link"])
        self.assertEqual([r["op"] for r in right["registers"]],
                         ["element", "element", "element_neg", "element_link"])

        # -[E] keeps the output pole iE but flips the input pole E -> -E.
        self.assertEqual(run(left)["matrix"][0][0],
                         {"re": -3.0, "im": -1.0})
        self.assertEqual(run(right)["matrix"][0][0],
                         {"re": 3.0, "im": 1.0})

    def test_element_negation_is_involutive_and_commutes_with_adjoint_state(self):
        twice = """fn main() -> Link<C[1],C[1]> {
            let $a = element(1 + i);
            return -(-$a);
        }"""
        program = optimize(compile_source(twice))
        self.assertEqual([r["op"] for r in program["registers"]], ["element"])
        self.assertEqual(run(program)["matrix"][0][0],
                         {"re": 1.0, "im": 1.0})

        neg_adjoint = """fn main() -> Link<C[1],C[1]> {
            let $a = element(1 + i);
            return (-$a)†;
        }"""
        state = run(optimize(compile_source(neg_adjoint)))
        self.assertEqual(state["element_state"],
                         {"pole_sign": -1, "adjoint": True})


if __name__ == "__main__":
    unittest.main()
