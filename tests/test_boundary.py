"""Boundary codec and boundary reduction of compose chains."""
import random
import unittest
from linkc.boundary import (extract, restore, compose_factors, reduce_chain,
                            prefix_products, segment_via_boundary,
                            boundary_reduce, chain_product)
from linkc.compiler import compile_source, optimize
from linkc.vm import run, multiply

SRC_CANCEL = """
fn main() -> Link<C[2], C[2]> {
    let $B: Link<C[2], C[2]> = matrix::<C[2], C[2]>([[1, 2], [0, 1]]);
    let $Binv: Link<C[2], C[2]> = matrix::<C[2], C[2]>([[1, -2], [0, 1]]);
    let $A: Link<C[2], C[2]> = matrix::<C[2], C[2]>([[3, 1], [0, 2]]);
    return $A @ $Binv @ $B;
}
"""

def as_floats(m):
    return [[z['re'] for z in row] for row in m]

class CodecTests(unittest.TestCase):
    def test_sign_stream_switches(self):
        s = [1] * 100 + [-1] * 50 + [1] * 200
        initial, switches = extract(s)
        self.assertEqual(initial, 1)
        self.assertEqual(len(switches), 2)  # boundary form: 1 + 2 numbers
        self.assertEqual(restore(initial, switches, len(s)), s)

    def test_random_piecewise_constant_roundtrip(self):
        rng = random.Random(7)
        for _ in range(200):
            s = []
            v = rng.randint(-5, 5)
            for _ in range(rng.randint(1, 60)):
                if rng.random() < 0.1: v = rng.randint(-5, 5)
                s.append(v)
            initial, switches = extract(s)
            self.assertEqual(restore(initial, switches, len(s)), s)

class ChainReductionTests(unittest.TestCase):
    def test_inverse_pair_cancels_at_seam(self):
        prog = optimize(compile_source(SRC_CANCEL))
        factors = compose_factors(prog)
        self.assertEqual(len(factors), 3)
        reduced, stats = reduce_chain(factors)
        self.assertEqual(stats['cancelled_pairs'], 1)
        self.assertEqual(len(reduced), 1)
        self.assertEqual(reduced[0], [[3 + 0j, 1 + 0j], [0j, 2 + 0j]])

    def test_synthesized_program_equivalent(self):
        prog = optimize(compile_source(SRC_CANCEL))
        report = boundary_reduce(prog)
        self.assertIsNotNone(report['program'])
        self.assertEqual(len(report['program']['registers']), 1)
        self.assertEqual(as_floats(run(report['program'])['matrix']),
                         as_floats(run(prog)['matrix']))

    def test_power_of_two_scaling_cancels_exactly(self):
        big = [[2.0 ** 40, 0], [0, 2.0 ** 40]]
        small = [[2.0 ** -40, 0], [0, 2.0 ** -40]]
        A = [[1 + 0j, 1 + 0j], [0j, 1 + 0j]]
        reduced, stats = reduce_chain([big, small, A])
        self.assertEqual(stats['cancelled_pairs'], 1)
        self.assertEqual(reduced, [A])

    def test_non_exact_pair_is_not_cancelled(self):
        # Strict rule: a pair off from the exact inverse by one ulp must
        # not cancel. (1e8 * 1e-8 rounds to exactly 1.0, so use a shear
        # pair perturbed in the last bit instead.)
        B = [[1 + 0j, 2 + 0j], [0j, 1 + 0j]]
        almost_inv = [[1 + 0j, -2 + 0j], [0j, 1.0000000000000002 + 0j]]
        _reduced, stats = reduce_chain([B, almost_inv])
        self.assertEqual(stats['cancelled_pairs'], 0)

class BoundaryFormulaTests(unittest.TestCase):
    def test_segment_from_boundaries_exact(self):
        rng = random.Random(3)
        def shear():
            M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            i, j = rng.sample(range(3), 2)
            M[i][j] = rng.randint(-3, 3)
            return [[complex(v) for v in row] for row in M]
        factors = [shear() for _ in range(5)]
        lambdas = prefix_products(factors)
        self.assertEqual(lambdas[-1], chain_product(factors))
        seg = segment_via_boundary(factors, 2, 4)
        direct = chain_product(factors[1:4])
        self.assertEqual(seg, [[float(z.real) for z in row] for row in direct])

if __name__ == '__main__':
    unittest.main()
