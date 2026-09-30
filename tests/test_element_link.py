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


if __name__ == "__main__":
    unittest.main()
