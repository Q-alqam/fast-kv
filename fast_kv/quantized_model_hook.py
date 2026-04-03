"""Fast-KV quantized model hook for larger models.

Extends FastKVModelHook to support loading models with 4-bit weight
quantization via bitsandbytes. This allows running 7B-class models
on consumer hardware (~6-8 GB RAM).

Important: 4-bit quantization applies ONLY to model weights, not to
the KV cache. Fast-KV still operates on full float32 KV vectors.
The two compression techniques are independent and complementary.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("PyTorch and Transformers are required.")

try:
    import psutil
except ImportError:
    psutil = None

from fast_kv.config import FastKVConfig
from fast_kv.model_hook import FastKVModelHook

logger = logging.getLogger(__name__)


def _check_available_ram_gb() -> float:
    """Return available system RAM in GB."""
    if psutil is None:
        return 0.0
    return psutil.virtual_memory().available / (1024 ** 3)


def _check_cuda_available() -> bool:
    """Check if CUDA GPU is available for bitsandbytes."""
    return torch.cuda.is_available()


def select_best_model(available_gb: Optional[float] = None) -> str:
    """Select the best model that fits in available RAM.

    Args:
        available_gb: Override available RAM (auto-detected if None).

    Returns:
        HuggingFace model identifier string.
    """
    if available_gb is None:
        available_gb = _check_available_ram_gb()

    if available_gb >= 12:
        return "meta-llama/Llama-3.2-3B-Instruct"
    elif available_gb >= 6:
        return "microsoft/phi-2"
    else:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class QuantizedFastKVModelHook(FastKVModelHook):
    """FastKVModelHook with optional 4-bit weight quantization.

    Loads model weights at 4-bit precision via bitsandbytes (if CUDA
    available) or at float32 with low_cpu_mem_usage (CPU fallback).
    Fast-KV KV cache compression runs at full float32 precision
    regardless of weight quantization.

    Args:
        model_name: HuggingFace model identifier.
        fast_kv_config: FastKVConfig instance.
        load_in_4bit: Whether to attempt 4-bit loading via bitsandbytes.
    """

    def __init__(
        self,
        model_name: str,
        fast_kv_config: Optional[FastKVConfig] = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.fast_kv_config = fast_kv_config or FastKVConfig()
        self.loading_method = "unknown"
        self._load_model(model_name, load_in_4bit)

    def _load_model(self, model_name: str, load_in_4bit: bool) -> None:
        """Load model with best available method.

        Tries 4-bit bitsandbytes first (requires CUDA), then falls back
        to float32 with low_cpu_mem_usage.
        """
        from fast_kv.model_hook import (
            _compute_kv_dim,
            _detect_attention_class,
            _get_attention_modules,
            _get_num_layers,
        )
        from fast_kv.fast_kv_cache import FastKVCache

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Try 4-bit loading if requested and CUDA available
        loaded = False
        if load_in_4bit and _check_cuda_available():
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Loading %s with 4-bit quantization...", model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self.loading_method = "4-bit (bitsandbytes NF4)"
                loaded = True
                logger.info("Loaded with 4-bit quantization")
            except Exception as e:
                logger.warning("4-bit loading failed: %s. Falling back.", e)

        # Fallback: float32 with low CPU memory usage
        if not loaded:
            logger.info("Loading %s with float32 (low_cpu_mem_usage)...", model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                self.loading_method = "float32 (CPU, low memory)"
                logger.info("Loaded with float32")
            except Exception as e:
                logger.error("Failed to load model: %s", e)
                raise

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

        # Detect architecture
        self.attention_class_name = _detect_attention_class(self.model)
        self._attention_modules = _get_attention_modules(self.model)

        # Extract model dimensions
        mc = self.model.config
        self._n_layers = _get_num_layers(mc)
        self._kv_dim = _compute_kv_dim(mc)
        self._num_attention_heads = mc.num_attention_heads
        self._head_dim = mc.hidden_size // mc.num_attention_heads
        self._num_kv_heads = getattr(mc, "num_key_value_heads", mc.num_attention_heads)

        # Initialize FastKVCache
        model_config_dict = {
            "n_layers": self._n_layers,
            "kv_dim": self._kv_dim,
            "dtype": "float32",
        }
        self.fast_kv_cache = FastKVCache(self.fast_kv_config, model_config_dict)

        # Tracking state
        self._generation_step = 0
        self._captured_attentions = {}
        self._hook_handles = []
        self.last_attention_weights = {0: []}
        self.generation_times = {}
        self.ram_measurements = {}

        # Register attention hooks
        self._register_attention_hooks()

        logger.info(
            "QuantizedFastKVModelHook: %s, %d layers, kv_dim=%d, method=%s",
            model_name, self._n_layers, self._kv_dim, self.loading_method,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model info including loading method."""
        info = super().get_model_info()
        info["loading_method"] = self.loading_method
        return info

    def __repr__(self) -> str:
        return (
            f"QuantizedFastKVModelHook(model={self.model_name!r}, "
            f"method={self.loading_method!r}, "
            f"layers={self._n_layers}, kv_dim={self._kv_dim})"
        )
