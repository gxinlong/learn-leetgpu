import os
import sys
import unittest

import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import benchmark_triton as bench


def gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
         M: int, N: int, K: int, alpha: float, beta: float = 0.0):
    pass


class HarnessTests(unittest.TestCase):
    def test_parameters_and_exact_rectangular_shapes(self):
        _, scalars, specs = bench.prepare_specs(
            gemm, {'M': '3', 'N': '5', 'K': '7', 'alpha': '-0.5'},
            {'a': 'MxK', 'b': 'K,N', 'c': 'MxN'}, {'c': 'float16'}, 'float32', 0)
        self.assertEqual(scalars['alpha'], -0.5)
        self.assertEqual(scalars['beta'], 0.0)
        self.assertEqual(specs, {'a': ((3, 7), 'float32'), 'b': ((7, 5), 'float32'), 'c': ((3, 5), 'float16')})

    def test_flat_fallback_and_override(self):
        raw = dict(M='3', N='5', K='7', alpha='1')
        self.assertEqual(bench.prepare_specs(gemm, raw, {}, {}, 'float32', 0)[2]['a'][0], (35,))
        self.assertEqual(bench.prepare_specs(gemm, raw, {}, {}, 'float32', 100)[2]['a'][0], (100,))

    def test_missing_and_unknown_arguments(self):
        with self.assertRaisesRegex(ValueError, 'Missing scalar'):
            bench.prepare_specs(gemm, {}, {}, {}, 'float32', 0)
        with self.assertRaisesRegex(ValueError, 'Unknown scalar'):
            bench.prepare_specs(gemm, {'typo': '1'}, {}, {}, 'float32', 0)
        with self.assertRaisesRegex(ValueError, 'unknown tensor'):
            bench.prepare_specs(gemm, {}, {'typo': '5'}, {}, 'float32', 0)

    def test_unknown_dtype(self):
        with self.assertRaisesRegex(ValueError, 'Unknown dtype'):
            bench.prepare_specs(gemm, dict(M='3', N='5', K='7', alpha='1'), {}, {'a': 'bad'}, 'float32', 0)

    def test_cli_scalar_forms(self):
        self.assertEqual(bench.parse_scalar_options(['--M=3', '--alpha', '-0.5'], ['gpu=2']),
                         {'M': '3', 'alpha': '-0.5', 'gpu': '2'})
        with self.assertRaises(ValueError):
            bench.parse_scalar_options(['--M=3'], ['M=4'])
        with self.assertRaises(ValueError):
            bench.parse_scalar_options(['--M'], [])

    def test_bad_shapes(self):
        for shape in ['0', '-2', '3x', '3xx4', 'missing', '__import__(os)']:
            with self.assertRaises(ValueError):
                bench.parse_shape(shape, {})

    def test_scalar_validation(self):
        self.assertFalse(bench.scalar_value('false', 'bool', 'flag'))
        self.assertTrue(bench.scalar_value('1', 'bool', 'flag'))
        for raw, kind in [('maybe', 'bool'), ('1.5', 'int'), ('nan', 'float')]:
            with self.assertRaises(ValueError):
                bench.scalar_value(raw, kind, 'arg')

    def test_keyword_only_defaults_and_string_annotations(self):
        def sample(a: 'torch.Tensor', N: 'int' = 4, *, flag: 'bool' = False):
            return a, N, flag
        signature, scalars, specs = bench.prepare_specs(sample, {}, {}, {}, 'float32', 0)
        self.assertEqual(specs['a'][0], (4,))
        self.assertEqual(bench.make_call(sample, signature, dict(a='sentinel', **scalars))(), ('sentinel', 4, False))

    def test_variadic_rejected(self):
        def bad(*args):
            pass
        with self.assertRaisesRegex(ValueError, 'Variadic'):
            bench.prepare_specs(bad, {}, {}, {}, 'float32', 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
