"""Quantization and dequantization engine for Fast-KV.

Implements uniform scalar quantization at arbitrary bit widths (1, 2, 4, 8, 16, 32)
with optional residual error storage for high-accuracy reconstruction.
"""

import logging
import time
from typing import Dict, Optional

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


def dequantize_vector(quantized_data: Dict) -> np.ndarray:
    """Dequantize a previously quantized vector back to float32.

    Reverses quantize_vector by applying: value = quantized * scale + zero_point

    Args:
        quantized_data: Dictionary returned by quantize_vector.

    Returns:
        Reconstructed float32 vector.
    """
    bits = quantized_data["bits"]
    scale = quantized_data["scale"]
    zero_point = quantized_data["zero_point"]
    shape = quantized_data["shape"]

    if bits == 32:
        return quantized_data["quantized"].copy()

    n_elements = int(np.prod(shape))

    if quantized_data["packed"]:
        raw = quantized_data["quantized"]
        if bits == 1:
            unpacked = np.unpackbits(raw)[:n_elements].astype(np.float32)
        elif bits == 2:
            # Unpack 4 values per byte
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
    return restored.reshape(shape)


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
