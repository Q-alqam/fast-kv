"""Tests for the compression engine."""

import pytest
import numpy as np

from fast_kv.compression import (
    apply_residual,
    benchmark_compression,
    compression_ratio,
    compute_residual,
    dequantize_vector,
    quantize_vector,
)


@pytest.fixture
def random_vector():
    """Create a random float32 vector."""
    np.random.seed(42)
    return np.random.randn(1024).astype(np.float32)


class TestQuantizeDequantize:
    """Tests for quantization roundtrip accuracy."""

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_roundtrip_shape_preserved(self, random_vector, bits):
        """Quantize + dequantize should preserve vector shape."""
        qdata = quantize_vector(random_vector, bits)
        restored = dequantize_vector(qdata)
        assert restored.shape == random_vector.shape

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_roundtrip_finite(self, random_vector, bits):
        """Reconstructed values should all be finite."""
        qdata = quantize_vector(random_vector, bits)
        restored = dequantize_vector(qdata)
        assert np.all(np.isfinite(restored))

    def test_higher_bits_lower_error(self, random_vector):
        """Higher bit widths should produce lower reconstruction error."""
        errors = {}
        for bits in [1, 2, 4, 8]:
            qdata = quantize_vector(random_vector, bits)
            restored = dequantize_vector(qdata)
            errors[bits] = np.abs(random_vector - restored).mean()

        assert errors[8] < errors[4] < errors[2] < errors[1]

    def test_32bit_is_lossless(self, random_vector):
        """32-bit quantization should be effectively lossless."""
        qdata = quantize_vector(random_vector, 32)
        restored = dequantize_vector(qdata)
        np.testing.assert_allclose(random_vector, restored, atol=1e-7)

    def test_8bit_low_error(self, random_vector):
        """8-bit quantization should have very low error."""
        qdata = quantize_vector(random_vector, 8)
        restored = dequantize_vector(qdata)
        mae = np.abs(random_vector - restored).mean()
        assert mae < 0.05  # Very low error for 8-bit


class TestEdgeCases:
    """Tests for edge cases in quantization."""

    def test_zero_vector(self):
        """Quantizing a zero vector should work without errors."""
        zero = np.zeros(128, dtype=np.float32)
        for bits in [1, 2, 4, 8]:
            qdata = quantize_vector(zero, bits)
            restored = dequantize_vector(qdata)
            assert restored.shape == zero.shape

    def test_constant_vector(self):
        """Quantizing a constant vector should work (zero range)."""
        const = np.full(128, 3.14, dtype=np.float32)
        for bits in [1, 2, 4, 8]:
            qdata = quantize_vector(const, bits)
            restored = dequantize_vector(qdata)
            assert restored.shape == const.shape

    def test_very_large_values(self):
        """Quantizing vectors with large magnitudes should work."""
        large = np.array([1e6, -1e6, 0.0, 5e5], dtype=np.float32)
        for bits in [1, 2, 4, 8]:
            qdata = quantize_vector(large, bits)
            restored = dequantize_vector(qdata)
            assert np.all(np.isfinite(restored))

    def test_single_element(self):
        """Quantizing a single-element vector should work."""
        single = np.array([0.5], dtype=np.float32)
        qdata = quantize_vector(single, 8)
        restored = dequantize_vector(qdata)
        assert restored.shape == (1,)


class TestResiduals:
    """Tests for residual error storage."""

    def test_residual_improves_accuracy(self, random_vector):
        """Applying residual correction should reduce reconstruction error."""
        qdata = quantize_vector(random_vector, 4)
        restored_no_res = dequantize_vector(qdata)
        error_no_res = np.abs(random_vector - restored_no_res).mean()

        residual = compute_residual(random_vector, qdata, residual_bits=8)
        restored_with_res = apply_residual(qdata, residual)
        error_with_res = np.abs(random_vector - restored_with_res).mean()

        assert error_with_res < error_no_res

    def test_residual_shape_preserved(self, random_vector):
        """Residual-corrected reconstruction should preserve shape."""
        qdata = quantize_vector(random_vector, 4)
        residual = compute_residual(random_vector, qdata, 8)
        restored = apply_residual(qdata, residual)
        assert restored.shape == random_vector.shape


class TestCompressionRatio:
    """Tests for compression ratio computation."""

    def test_expected_ratios(self, random_vector):
        """Compression ratios should be approximately 32/bits."""
        assert compression_ratio(random_vector, 1) == 32.0
        assert compression_ratio(random_vector, 2) == 16.0
        assert compression_ratio(random_vector, 4) == 8.0
        assert compression_ratio(random_vector, 8) == 4.0
        assert compression_ratio(random_vector, 32) == 1.0


class TestBenchmarkCompression:
    """Tests for the benchmark_compression utility."""

    def test_benchmark_returns_all_bit_widths(self, random_vector):
        """benchmark_compression should return results for all tested widths."""
        results = benchmark_compression(random_vector)
        assert set(results.keys()) == {1, 2, 4, 8}

    def test_benchmark_has_all_metrics(self, random_vector):
        """Each result should contain all expected metrics."""
        results = benchmark_compression(random_vector)
        expected_keys = {
            "compression_ratio", "mean_absolute_error", "max_absolute_error",
            "time_to_compress", "time_to_decompress",
        }
        for bits, metrics in results.items():
            assert set(metrics.keys()) == expected_keys
