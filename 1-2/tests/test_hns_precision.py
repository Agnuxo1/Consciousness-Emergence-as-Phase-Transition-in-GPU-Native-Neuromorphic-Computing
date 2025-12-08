"""
Test Suite for HNS (Hierarchical Numeric System) Precision

Tests the claim that HNS encoding provides superior precision:
- Documented error rate: 0.00×10⁰ (effectively zero)
- Expected improvement: ~2000-3000x over float32
"""

import pytest
import numpy as np
from utils.cpu_reference import hns_encode, hns_decode, hns_add


class TestHNSEncodeDecode:
    """Test HNS encoding and decoding."""

    def test_encode_decode_identity(self):
        """
        Test: decode(encode(x)) ≈ x
        """
        test_values = [0.0, 1.0, -1.0, 123.456, -987.654, 1e6, -1e9]

        for val in test_values:
            encoded = hns_encode(val)
            decoded = hns_decode(encoded)

            error = abs(decoded - val)
            assert error < 1e-6, \
                f"HNS encode/decode error for {val}: {error:.10f}"

    def test_encode_large_numbers(self):
        """
        Test HNS encoding of large numbers (up to billions).
        """
        test_values = [1e9, 5e9, 9.99e9, -3e9]

        for val in test_values:
            encoded = hns_encode(val)
            decoded = hns_decode(encoded)

            rel_error = abs(decoded - val) / abs(val)
            assert rel_error < 1e-6, \
                f"HNS large number error for {val}: {rel_error:.10f}"


class TestHNSPrecision:
    """Test HNS precision vs float32."""

    def test_accumulation_precision_hns_vs_float32(self):
        """
        Test: Accumulate 10,000 small additions.

        HNS should maintain precision much better than float32.
        """
        n_iterations = 10000
        small_value = 0.0001

        # Float32 accumulation
        sum_float32 = np.float32(0.0)
        for _ in range(n_iterations):
            sum_float32 += np.float32(small_value)

        expected = n_iterations * small_value
        error_float32 = abs(sum_float32 - expected)

        # HNS accumulation
        sum_hns = hns_encode(0.0)
        small_hns = hns_encode(small_value)
        for _ in range(n_iterations):
            sum_hns = hns_add(sum_hns, small_hns)

        decoded = hns_decode(sum_hns)
        error_hns = abs(decoded - expected)

        print(f"\nAccumulation test ({n_iterations} iterations):")
        print(f"  Float32 error: {error_float32:.10e}")
        print(f"  HNS error:     {error_hns:.10e}")
        print(f"  Improvement:   {error_float32 / (error_hns + 1e-20):.1f}x")

        # HNS should be significantly better
        assert error_hns < error_float32, \
            "HNS not more precise than float32"

        # HNS should be < 1e-10 as claimed
        assert error_hns < 1e-10, \
            f"HNS error {error_hns:.10e} exceeds claimed precision"

    def test_catastrophic_cancellation(self):
        """
        Test: (a + b) - a where b << a (catastrophic cancellation).

        This is a classic floating-point precision test.
        """
        a = 1e9
        b = 1.0

        # Float32
        result_float32 = (np.float32(a) + np.float32(b)) - np.float32(a)
        error_float32 = abs(result_float32 - b)

        # HNS
        a_hns = hns_encode(a)
        b_hns = hns_encode(b)
        sum_hns = hns_add(a_hns, b_hns)
        result_hns = hns_add(sum_hns, hns_encode(-a))
        decoded = hns_decode(result_hns)
        error_hns = abs(decoded - b)

        print(f"\nCatastrophic cancellation test:")
        print(f"  Float32 error: {error_float32:.10e}")
        print(f"  HNS error:     {error_hns:.10e}")

        # HNS should handle this better
        assert error_hns < error_float32 or error_hns < 1e-6
