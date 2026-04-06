"""Tests for the compression engine."""

import pytest
import numpy as np

from fast_kv.compression import (
    apply_residual,
    benchmark_compression,
    compression_ratio,
    compute_residual,
    detect_outliers,
    dequantize_vector,
    quantize_vector,
    quantize_vector_channelwise,
    quantize_vector_channelwise_outlier_aware,
    quantize_vector_outlier_aware,
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


class TestOutlierDetection:
    """Tests for outlier detection and outlier-aware quantization."""

    def test_detect_outliers_finds_clear_outliers(self):
        """Should detect obvious outliers in a vector."""
        np.random.seed(42)
        # Need enough normal values so the outlier is statistically extreme
        vector = np.random.randn(100).astype(np.float32) * 0.3
        vector[42] = 500.0  # Clear outlier at index 42
        result = detect_outliers(vector)
        assert 42 in result["outlier_indices"]
        assert len(result["outlier_values"]) > 0

    def test_detect_outliers_clean_vector(self):
        """Clean normal vectors should have no outliers."""
        np.random.seed(99)
        vector = np.random.randn(1024).astype(np.float32) * 0.5
        result = detect_outliers(vector)
        # With 3-sigma threshold, ~0.3% chance per element; a few might trigger
        # but it should be very few
        assert len(result["outlier_indices"]) < 20

    def test_outlier_aware_better_mae_than_standard(self):
        """Outlier-aware quantization must have lower MAE with outliers present."""
        np.random.seed(42)
        vector = np.random.randn(1024).astype(np.float32) * 0.3
        vector[42] = 500.0
        vector[300] = -480.0

        standard = quantize_vector(vector, 4)
        outlier_aware = quantize_vector_outlier_aware(vector, 4)

        std_recon = dequantize_vector(standard)
        oa_recon = dequantize_vector(outlier_aware)

        std_mae = np.abs(vector - std_recon).mean()
        oa_mae = np.abs(vector - oa_recon).mean()

        assert oa_mae < std_mae

    def test_outlier_aware_perfect_outlier_reconstruction(self):
        """Outlier values must be perfectly reconstructed."""
        np.random.seed(42)
        vector = np.random.randn(512).astype(np.float32)
        vector[10] = 999.0

        compressed = quantize_vector_outlier_aware(vector, 4)
        reconstructed = dequantize_vector(compressed)

        assert abs(reconstructed[10] - 999.0) < 0.001

    def test_backward_compatibility(self):
        """Old format without outlier fields must still dequantize."""
        np.random.seed(42)
        vector = np.random.randn(256).astype(np.float32)
        old_format = quantize_vector(vector, 4)
        reconstructed = dequantize_vector(old_format)
        assert reconstructed.shape == vector.shape

    def test_outlier_aware_roundtrip_no_outliers(self):
        """Uniform vector with no extreme values should produce has_outliers=False."""
        # Use uniform distribution to avoid any sigma-based outliers
        np.random.seed(42)
        vector = np.random.uniform(-0.1, 0.1, size=256).astype(np.float32)
        result = quantize_vector_outlier_aware(vector, 4, threshold_sigma=4.0)
        assert result["has_outliers"] is False


class TestChannelWise:
    """Tests for channel-wise quantization."""

    def test_channelwise_better_mae_than_scalar(self):
        """Channel-wise must have lower MAE on vectors with varied ranges."""
        np.random.seed(42)
        vector = np.zeros(1024, dtype=np.float32)
        vector[:256] = np.random.randn(256) * 0.01   # tiny range
        vector[256:512] = np.random.randn(256) * 10.0 # large range
        vector[512:768] = np.random.randn(256) * 0.1  # medium
        vector[768:] = np.random.randn(256) * 50.0    # very large

        scalar_q = quantize_vector(vector, 4)
        cw_q = quantize_vector_channelwise(vector, 4, group_size=64)

        scalar_mae = np.abs(vector - dequantize_vector(scalar_q)).mean()
        cw_mae = np.abs(vector - dequantize_vector(cw_q)).mean()

        assert cw_mae < scalar_mae

    def test_channelwise_group_size_tradeoff(self):
        """Smaller group size should give better MAE."""
        np.random.seed(42)
        vector = np.random.randn(1024).astype(np.float32)
        vector[::4] *= 20.0  # every 4th dim is large

        results = {}
        for gs in [128, 64, 32]:
            q = quantize_vector_channelwise(vector, 4, group_size=gs)
            r = dequantize_vector(q)
            results[gs] = np.abs(vector - r).mean()

        assert results[32] < results[64] < results[128]

    def test_channelwise_backward_compatible(self):
        """Old scalar format must still dequantize via router."""
        np.random.seed(42)
        vector = np.random.randn(256).astype(np.float32)
        old_format = quantize_vector(vector, 4)
        assert old_format.get("method", "scalar") == "scalar"
        reconstructed = dequantize_vector(old_format)
        assert reconstructed.shape == vector.shape

    def test_channelwise_preserves_per_dim_statistics(self):
        """8-bit per-dim channel-wise should be near-lossless."""
        np.random.seed(42)
        vector = np.random.randn(512).astype(np.float32)
        vector *= np.random.exponential(5.0, 512).astype(np.float32)

        q = quantize_vector_channelwise(vector, 8, group_size=1)
        r = dequantize_vector(q)
        mae = np.abs(vector - r).mean()
        assert mae < 0.5  # very tight with 8-bit per-dim

    def test_combined_channelwise_outlier_aware(self):
        """Channel-wise + outlier detection: outliers perfectly reconstructed."""
        np.random.seed(42)
        vector = np.random.randn(1024).astype(np.float32) * 0.5
        vector[50] = 500.0
        vector[200] = -480.0

        result = quantize_vector_channelwise_outlier_aware(
            vector, 4, group_size=64, threshold_sigma=3.0
        )
        reconstructed = dequantize_vector(result)

        assert abs(reconstructed[50] - 500.0) < 0.001
        assert abs(reconstructed[200] - (-480.0)) < 0.001

        normal_mask = np.ones(1024, dtype=bool)
        normal_mask[[50, 200]] = False
        normal_mae = np.abs(vector[normal_mask] - reconstructed[normal_mask]).mean()
        assert normal_mae < 0.5
