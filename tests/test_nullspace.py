"""Null-space carry: exact ledger tests (Gaussian-rational core)."""
import unittest
from fractions import Fraction
from linkc.nullspace import (kernel_basis, nullity, carry_decompose, restore,
                             apply_with_carry, _rat_matrix, _apply_exact, _zero)

def matvec(x, A):
    return [sum(x[i] * A[i][j] for i in range(len(x))) for j in range(len(A[0]))]

class KernelTests(unittest.TestCase):
    def test_rank_deficient_2x2(self):
        A = [[1, 2], [2, 4]]  # x @ A = 0  <=>  x ~ (-2, 1)
        self.assertEqual(nullity(A), 1)
        (k,) = kernel_basis(A)
        self.assertEqual(matvec(k, A), [0, 0])
        self.assertAlmostEqual(k[0] / k[1], -2.0)

    def test_full_rank_has_no_kernel(self):
        self.assertEqual(nullity([[1, 0], [0, 1]]), 0)
        self.assertEqual(nullity([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 0)

    def test_kernel_3x2(self):
        A = [[1, 0], [0, 1], [1, 1]]  # kernel spanned by (-1, -1, 1)
        self.assertEqual(nullity(A), 1)
        (k,) = kernel_basis(A)
        self.assertEqual(matvec(k, A), [0, 0])

class CarryTests(unittest.TestCase):
    def test_integral_case_exact_floats(self):
        A = [[1, 2], [2, 4]]
        x = [3, 1]
        y, decomp = apply_with_carry(A, x)
        self.assertEqual(y, matvec(x, A))            # (5, 10)
        self.assertEqual(matvec(decomp['kernel'], A), [0, 0])
        self.assertEqual(restore(decomp), [3, 1])    # recycle box: nothing lost

    def test_pure_kernel_input(self):
        A = [[1, 0], [0, 1], [1, 1]]
        x = [-1, -1, 1]  # entirely in ker A
        y, decomp = apply_with_carry(A, x)
        self.assertEqual(y, [0, 0])
        self.assertEqual(decomp['range'], [0, 0, 0])
        self.assertEqual(restore(decomp), x)

    def test_exact_ledger_random_rationals(self):
        # The ledger must close over Q(i), not just approximately.
        A = [[Fraction(1, 2), 3, 0], [2, Fraction(-1, 3), 1], [1, 1, 1], [0, 2, Fraction(2, 5)]]
        x = [Fraction(3, 7), -2, Fraction(1, 3), 5]
        Ar = _rat_matrix(A)
        decomp = carry_decompose(A, x)  # exact Fraction inputs end to end
        ex = decomp['exact']
        self.assertTrue(all(_zero(z) for z in _apply_exact(Ar, ex['kernel'])))
        summed = [(a[0] + b[0], a[1] + b[1]) for a, b in zip(ex['range'], ex['kernel'])]
        self.assertEqual(summed, [(Fraction(v), Fraction(0)) for v in x])
        y_exact = _apply_exact(Ar, ex['range'])
        self.assertEqual(y_exact, _apply_exact(Ar, [(Fraction(v), Fraction(0)) for v in x]))

if __name__ == '__main__':
    unittest.main()
