"""Quantization and dequantization engine for Fast-KV.

Implements three quantization methods:
  1. Uniform scalar: one scale for the whole vector (fast, low quality)
  2. Outlier-aware scalar: outliers stored separately (moderate quality)
  3. Channel-wise: separate scale per group of dimensions (best quality)

All methods support arbitrary bit widths (1, 2, 4, 8, 16, 32) and are
dispatched transparently through the dequantize_vector() router.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Map bit widths to the smallest numpy dtype that can hold quantized values
_DTYPE_MAP = {
    1: np.uint8,
    2: np.uint8,
    4: np.uint8,
    8: np.uint8,
    16: np.uint16,
    32: np.float32,
}


def quantize_vector(vector: np.ndarray, bits: int) -> Dict:
    """Quantize a float32 vector to the specified bit width.

    Uses uniform scalar quantization: maps the vector's [min, max] range
    to [0, 2^bits - 1] integer levels.

    Args:
        vector: Input float32 vector to quantize.
        bits: Target bit width (1, 2, 4, 8, 16, or 32).

    Returns:
        Dictionary with keys:
            'quantized': np.ndarray of quantized integer values
            'scale': float scale factor for dequantization
            'zero_point': float zero point (v_min)
            'bits': int bit width used
            'shape': original vector shape
            'packed': bool, whether values are bit-packed
    """
    if bits == 32:
        return {
            "quantized": vector.copy(),
            "scale": 1.0,
            "zero_point": 0.0,
            "bits": 32,
            "shape": vector.shape,
            "packed": False,
        }

    v_min = float(vector.min())
    v_max = float(vector.max())
    n_levels = (1 << bits) - 1  # 2^bits - 1

    # Handle zero-range edge case (constant vector)
    if v_max - v_min < 1e-10:
        scale = 1.0
    else:
        scale = (v_max - v_min) / n_levels

    zero_point = v_min
    quantized = np.clip(
        np.round((vector - zero_point) / scale), 0, n_levels
    ).astype(np.uint8 if bits <= 8 else np.uint16)

    result = {
        "quantized": quantized,
        "scale": scale,
        "zero_point": zero_point,
        "bits": bits,
        "shape": vector.shape,
        "packed": False,
    }

    # Bit-pack for 1-bit and 2-bit to save memory
    if bits == 1:
        result["quantized"] = np.packbits(quantized.astype(np.uint8))
        result["packed"] = True
    elif bits == 2:
        # Pack 4 values per byte
        flat = quantized.flatten().astype(np.uint8)
        # Pad to multiple of 4
        pad_len = (4 - len(flat) % 4) % 4
        if pad_len:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])
        packed = (
            (flat[0::4] << 6)
            | (flat[1::4] << 4)
            | (flat[2::4] << 2)
            | flat[3::4]
        )
        result["quantized"] = packed.astype(np.uint8)
        result["packed"] = True
    elif bits == 4:
        # Pack 2 values per byte
        flat = quantized.flatten().astype(np.uint8)
        pad_len = len(flat) % 2
        if pad_len:
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2] << 4) | flat[1::2]
        result["quantized"] = packed.astype(np.uint8)
        result["packed"] = True

    return result


# ---------------------------------------------------------------------------
# Outlier-aware quantization
# ---------------------------------------------------------------------------


def detect_outliers(
    vector: np.ndarray, threshold_sigma: float = 3.0
) -> Dict:
    """Detect outlier values in a vector using sigma thresholding.

    A value is an outlier if it lies more than threshold_sigma standard
    deviations from the mean.

    Args:
        vector: Input float32 vector.
        threshold_sigma: Number of standard deviations for the cutoff.

    Returns:
        Dictionary with:
            'outlier_indices': list of int indices of outlier values.
            'outlier_values': list of float outlier values.
            'normal_mask': boolean np.ndarray (True for normal values).
    """
    mean = float(vector.mean())
    std = float(vector.std())

    if std < 1e-10:
        return {
            "outlier_indices": [],
            "outlier_values": [],
            "normal_mask": np.ones(len(vector), dtype=bool),
        }

    deviation = np.abs(vector - mean)
    outlier_mask = deviation > threshold_sigma * std
    indices = np.where(outlier_mask)[0].tolist()
    values = vector[outlier_mask].tolist()

    return {
        "outlier_indices": indices,
        "outlier_values": values,
        "normal_mask": ~outlier_mask,
    }


def quantize_vector_outlier_aware(
    vector: np.ndarray, bits: int, threshold_sigma: float = 3.0
) -> Dict:
    """Quantize a vector with outlier-aware handling.

    Detects outliers, stores them separately at full precision, and
    quantizes the remaining normal values with a tighter scale.

    Args:
        vector: Input float32 vector.
        bits: Target bit width.
        threshold_sigma: Sigma threshold for outlier detection.

    Returns:
        Extended quantized dict with outlier information.
    """
    outlier_info = detect_outliers(vector, threshold_sigma)
    outlier_indices = outlier_info["outlier_indices"]

    if not outlier_indices:
        # No outliers — use standard quantization, mark as clean
        result = quantize_vector(vector, bits)
        result["has_outliers"] = False
        result["outliers"] = []
        result["outlier_count"] = 0
        result["vector_length"] = len(vector)
        return result

    # Zero out outlier positions and quantize the rest
    clean_vector = vector.copy()
    clean_vector[outlier_indices] = 0.0

    # Compute scale from normal values only for better precision
    normal_mask = outlier_info["normal_mask"]
    normal_vals = vector[normal_mask]
    if len(normal_vals) > 0:
        clean_vector[outlier_indices] = float(normal_vals.mean())

    result = quantize_vector(clean_vector, bits)
    result["has_outliers"] = True
    result["outliers"] = list(zip(outlier_indices, outlier_info["outlier_values"]))
    result["outlier_count"] = len(outlier_indices)
    result["vector_length"] = len(vector)

    return result


# ---------------------------------------------------------------------------
# Channel-wise quantization
# ---------------------------------------------------------------------------


def quantize_vector_channelwise(
    vector: np.ndarray, bits: int, group_size: Optional[int] = 64
) -> Dict:
    """Quantize a vector with per-group scales (channel-wise).

    Instead of one scale for the entire vector, computes a separate
    scale and zero point for each group of dimensions. This preserves
    per-channel statistical structure that attention depends on.

    Args:
        vector: Input float32 vector of shape (dim,).
        bits: Target bit width (1, 2, 4, 8).
        group_size: Dimensions per quantization group. None = per-dimension.

    Returns:
        Quantized dict with 'method'='channelwise', 'scales' array,
        'zero_points' array, and 'group_size'.
    """
    dim = len(vector)
    n_levels = (1 << bits) - 1  # 2^bits - 1

    if group_size is None:
        # Per-dimension quantization
        n_groups = dim
        scales = np.zeros(dim, dtype=np.float32)
        zero_points = np.zeros(dim, dtype=np.float32)
        quantized = np.zeros(dim, dtype=np.uint8 if bits <= 8 else np.uint16)

        for d in range(dim):
            v_min = float(vector[d])
            v_max = float(vector[d])
            # Add tiny range so single values don't get scale=0
            rng = max(abs(v_max - v_min), 1e-8)
            v_min -= rng * 0.5
            v_max += rng * 0.5
            scales[d] = (v_max - v_min) / n_levels
            zero_points[d] = v_min
            quantized[d] = int(np.clip(
                round((vector[d] - v_min) / scales[d]), 0, n_levels
            ))
    else:
        # Group-wise quantization
        n_groups = math.ceil(dim / group_size)
        scales = np.zeros(n_groups, dtype=np.float32)
        zero_points = np.zeros(n_groups, dtype=np.float32)
        quantized = np.zeros(dim, dtype=np.uint8 if bits <= 8 else np.uint16)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, dim)
            group = vector[start:end]

            v_min = float(group.min())
            v_max = float(group.max())
            if v_max - v_min < 1e-10:
                scales[g] = 1.0
            else:
                scales[g] = (v_max - v_min) / n_levels
            zero_points[g] = v_min

            quantized[start:end] = np.clip(
                np.round((group - v_min) / scales[g]), 0, n_levels
            ).astype(quantized.dtype)

    return {
        "quantized": quantized,
        "scales": scales,
        "zero_points": zero_points,
        "bits": int(bits),
        "dim": dim,
        "group_size": group_size,
        "method": "channelwise",
        "has_outliers": False,
        "outliers": [],
        "outlier_count": 0,
    }


def quantize_vector_channelwise_outlier_aware(
    vector: np.ndarray,
    bits: int,
    group_size: int = 64,
    threshold_sigma: float = 3.0,
) -> Dict:
    """Channel-wise quantization with outlier detection.

    Combines outlier-aware handling with channel-wise quantization
    for maximum reconstruction quality.

    Args:
        vector: Input float32 vector.
        bits: Target bit width.
        group_size: Dimensions per quantization group.
        threshold_sigma: Sigma threshold for outlier detection.

    Returns:
        Channel-wise quantized dict with outlier information.
    """
    outlier_info = detect_outliers(vector, threshold_sigma)
    outlier_indices = outlier_info["outlier_indices"]

    if not outlier_indices:
        result = quantize_vector_channelwise(vector, bits, group_size)
        return result

    # Replace outliers with group mean for cleaner quantization
    clean_vector = vector.copy()
    normal_mask = outlier_info["normal_mask"]
    normal_mean = float(vector[normal_mask].mean()) if normal_mask.any() else 0.0
    clean_vector[outlier_indices] = normal_mean

    result = quantize_vector_channelwise(clean_vector, bits, group_size)
    result["has_outliers"] = True
    result["outliers"] = list(zip(outlier_indices, outlier_info["outlier_values"]))
    result["outlier_count"] = len(outlier_indices)

    return result


def _dequantize_scalar(quantized_data: Dict) -> np.ndarray:
    """Dequantize a scalar-quantized vector back to float32.

    Internal implementation for uniform scalar quantization.
    """
    bits = quantized_data["bits"]
    scale = quantized_data["scale"]
    zero_point = quantized_data["zero_point"]
    shape = quantized_data["shape"]

    if bits == 32:
        return quantized_data["quantized"].copy()

    n_elements = int(np.prod(shape))

    if quantized_data.get("packed", False):
        raw = quantized_data["quantized"]
        if bits == 1:
            unpacked = np.unpackbits(raw)[:n_elements].astype(np.float32)
        elif bits == 2:
            unpacked = np.zeros(len(raw) * 4, dtype=np.uint8)
            unpacked[0::4] = (raw >> 6) & 0x03
            unpacked[1::4] = (raw >> 4) & 0x03
            unpacked[2::4] = (raw >> 2) & 0x03
            unpacked[3::4] = raw & 0x03
            unpacked = unpacked[:n_elements].astype(np.float32)
        elif bits == 4:
            unpacked = np.zeros(len(raw) * 2, dtype=np.uint8)
            unpacked[0::2] = (raw >> 4) & 0x0F
            unpacked[1::2] = raw & 0x0F
            unpacked = unpacked[:n_elements].astype(np.float32)
        else:
            unpacked = raw.astype(np.float32)
    else:
        unpacked = quantized_data["quantized"].astype(np.float32)

    restored = unpacked * scale + zero_point
    restored = restored.reshape(shape)

    # Restore outlier values if present
    if quantized_data.get("has_outliers") and quantized_data.get("outliers"):
        flat = restored.flatten()
        for idx, val in quantized_data["outliers"]:
            if idx < len(flat):
                flat[idx] = val
        restored = flat.reshape(shape)

    return restored


def _dequantize_channelwise(quantized_data: Dict) -> np.ndarray:
    """Dequantize a channel-wise quantized vector back to float32.

    Each group of dimensions has its own scale and zero point.
    """
    quantized = quantized_data["quantized"]
    scales = quantized_data["scales"]
    zero_points = quantized_data["zero_points"]
    group_size = quantized_data["group_size"]
    dim = quantized_data["dim"]

    reconstructed = np.zeros(dim, dtype=np.float32)

    if group_size is None:
        # Per-dimension: scales and zero_points are per-element
        reconstructed = quantized.astype(np.float32) * scales + zero_points
    else:
        n_groups = len(scales)
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, dim)
            reconstructed[start:end] = (
                quantized[start:end].astype(np.float32) * scales[g] + zero_points[g]
            )

    # Restore outlier values if present
    if quantized_data.get("has_outliers") and quantized_data.get("outliers"):
        for idx, val in quantized_data["outliers"]:
            if idx < dim:
                reconstructed[idx] = val

    return reconstructed


def dequantize_vector(quantized_data: Dict) -> np.ndarray:
    """Dequantize a quantized vector back to float32.

    Routes to the appropriate implementation based on the 'method' field.
    Backward compatible: old format without 'method' uses scalar path.

    Args:
        quantized_data: Dictionary returned by any quantize function.

    Returns:
        Reconstructed float32 vector.
    """
    method = quantized_data.get("method", "scalar")
    if method == "channelwise":
        return _dequantize_channelwise(quantized_data)
    else:
        return _dequantize_scalar(quantized_data)


def compute_residual(
    original: np.ndarray, quantized_data: Dict, residual_bits: int = 8
) -> Dict:
    """Compute and quantize the residual error from quantization.

    residual = original - dequantize(quantized)
    The residual itself is then quantized at residual_bits precision.

    Args:
        original: The original float32 vector before quantization.
        quantized_data: The quantized data dictionary.
        residual_bits: Bit width for quantizing the residual (default 8).

    Returns:
        Quantized residual dictionary (same format as quantize_vector output).
    """
    reconstructed = dequantize_vector(quantized_data)
    residual = original - reconstructed
    return quantize_vector(residual, residual_bits)


def apply_residual(quantized_data: Dict, residual_data: Dict) -> np.ndarray:
    """Reconstruct a vector using both quantized data and residual correction.

    result = dequantize(quantized) + dequantize(residual)

    Args:
        quantized_data: Primary quantized data dictionary.
        residual_data: Residual quantized data dictionary.

    Returns:
        Reconstructed float32 vector with residual correction applied.
    """
    base = dequantize_vector(quantized_data)
    res = dequantize_vector(residual_data)
    return base + res


def compression_ratio(original_vector: np.ndarray, bits: int) -> float:
    """Calculate the approximate compression ratio for a given bit width.

    Args:
        original_vector: The original float32 vector (used only for shape).
        bits: Target bit width.

    Returns:
        Compression ratio (e.g., 32/4 = 8.0 for 4-bit quantization).
    """
    if bits == 0:
        return float("inf")
    return 32.0 / bits


def benchmark_compression(vector: np.ndarray) -> Dict:
    """Benchmark quantization quality and speed across multiple bit widths.

    Runs quantize + dequantize for bits in [1, 2, 4, 8] and measures
    compression ratio, accuracy, and timing.

    Args:
        vector: Input float32 vector to benchmark.

    Returns:
        Dictionary mapping bit width to metrics:
            compression_ratio, mean_absolute_error, max_absolute_error,
            time_to_compress, time_to_decompress.
    """
    results = {}
    for bits in [1, 2, 4, 8]:
        # Time compression
        t0 = time.perf_counter()
        qdata = quantize_vector(vector, bits)
        t_compress = time.perf_counter() - t0

        # Time decompression
        t0 = time.perf_counter()
        restored = dequantize_vector(qdata)
        t_decompress = time.perf_counter() - t0

        error = np.abs(vector - restored)
        results[bits] = {
            "compression_ratio": compression_ratio(vector, bits),
            "mean_absolute_error": float(error.mean()),
            "max_absolute_error": float(error.max()),
            "time_to_compress": t_compress,
            "time_to_decompress": t_decompress,
        }
    return results


def benchmark_outlier_aware(vector: np.ndarray, bits: int = 4) -> Dict:
    """Compare standard vs outlier-aware quantization on a single vector.

    Args:
        vector: Input float32 vector (ideally one with outliers).
        bits: Bit width to test.

    Returns:
        Comparison dict with MAE, improvement, and outlier stats.
    """
    # Standard
    std_q = quantize_vector(vector, bits)
    std_r = dequantize_vector(std_q)
    std_mae = float(np.abs(vector - std_r).mean())

    # Outlier-aware
    oa_q = quantize_vector_outlier_aware(vector, bits)
    oa_r = dequantize_vector(oa_q)
    oa_mae = float(np.abs(vector - oa_r).mean())

    improvement = ((std_mae - oa_mae) / std_mae * 100) if std_mae > 0 else 0.0

    # Effective compression ratio accounting for outlier storage overhead
    n = len(vector)
    n_outliers = oa_q.get("outlier_count", 0)
    # Outliers cost 4 bytes (float32) + 4 bytes (index) = 8 bytes each
    outlier_bytes = n_outliers * 8
    compressed_bits = n * bits + outlier_bytes * 8
    original_bits = n * 32
    oa_cr = original_bits / compressed_bits if compressed_bits > 0 else 1.0

    return {
        "standard_mae": std_mae,
        "outlier_aware_mae": oa_mae,
        "improvement_percent": improvement,
        "outlier_count": n_outliers,
        "outlier_percent": n_outliers / n * 100 if n > 0 else 0,
        "compression_ratio_standard": 32.0 / bits,
        "compression_ratio_outlier_aware": oa_cr,
    }


def compare_quantization_methods(vector: np.ndarray, bits: int = 4) -> Dict:
    """Compare all quantization methods on a single vector.

    Args:
        vector: Input float32 vector.
        bits: Bit width to test.

    Returns:
        Dictionary mapping method name to {mae, cosine_sim, time}.
    """
    results = {}
    methods = {
        "scalar": lambda v, b: quantize_vector(v, b),
        "channelwise_128": lambda v, b: quantize_vector_channelwise(v, b, 128),
        "channelwise_64": lambda v, b: quantize_vector_channelwise(v, b, 64),
        "channelwise_32": lambda v, b: quantize_vector_channelwise(v, b, 32),
    }

    for name, quant_fn in methods.items():
        t0 = time.perf_counter()
        qdata = quant_fn(vector, bits)
        t_compress = time.perf_counter() - t0

        restored = dequantize_vector(qdata)
        error = np.abs(vector - restored)

        dot = np.dot(vector, restored)
        nv = np.linalg.norm(vector)
        nr = np.linalg.norm(restored)
        cosine = dot / (nv * nr) if nv > 0 and nr > 0 else 1.0

        results[name] = {
            "mae": float(error.mean()),
            "cosine_sim": float(cosine),
            "time": t_compress,
        }

    return results
