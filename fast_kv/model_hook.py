"""Fast-KV Phase 2: Real HuggingFace model integration.

Provides FastKVModelHook which intercepts a HuggingFace causal LM's KV cache
during token-by-token generation, routing key/value tensors through the
Fast-KV two-tier compression system.

The core loop:
  1. Run one forward pass through the model with use_cache=True.
  2. Intercept the resulting past_key_values (per-layer K/V tensors).
  3. Convert to numpy, push through FastKVCache for tier assignment
     and compression.
  4. Reconstruct past_key_values from FastKVCache (hot at full precision,
     cold decompressed from quantized form).
  5. Feed the reconstructed cache into the next forward pass.

This injects lossy reconstruction of cold-tier tokens back into the model,
simulating real-world KV cache compression during autoregressive generation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for FastKVModelHook. "
        "Install it with: pip install torch"
    )

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "Transformers is required for FastKVModelHook. "
        "Install it with: pip install transformers"
    )

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from fast_kv.config import FastKVConfig
from fast_kv.fast_kv_cache import FastKVCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

# Known attention module class names across popular architectures.
_ATTENTION_CLASS_NAMES: Dict[str, List[str]] = {
    "llama": ["LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2"],
    "phi": ["PhiAttention", "PhiSdpaAttention", "PhiFlashAttention2"],
    "phi3": ["Phi3Attention", "Phi3SdpaAttention", "Phi3FlashAttention2"],
    "gpt2": ["GPT2Attention"],
    "mistral": ["MistralAttention", "MistralSdpaAttention"],
    "gemma": ["GemmaAttention", "GemmaSdpaAttention"],
    "qwen2": ["Qwen2Attention", "Qwen2SdpaAttention"],
    "gpt_neox": ["GPTNeoXAttention"],
    "opt": ["OPTAttention"],
}


def _detect_attention_class(model: "transformers.PreTrainedModel") -> Optional[str]:
    """Inspect the model's layers and return the attention module class name.

    Walks the model's named modules looking for a class whose name matches
    one of the known attention module names. Returns the first match.

    Args:
        model: A HuggingFace PreTrainedModel instance.

    Returns:
        The class name string of the detected attention module, or None if
        no known attention class was found.
    """
    known_names: set = set()
    for names in _ATTENTION_CLASS_NAMES.values():
        known_names.update(names)

    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in known_names:
            logger.info("Detected attention class: %s", cls_name)
            return cls_name

    # Fallback: look for any module whose name contains "Attention"
    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Attention" in cls_name:
            logger.info(
                "Detected attention class (fallback heuristic): %s", cls_name
            )
            return cls_name

    logger.warning("Could not detect attention class for model")
    return None


def _get_attention_modules(
    model: "transformers.PreTrainedModel",
) -> List[Tuple[str, "torch.nn.Module"]]:
    """Return a list of (name, module) pairs for all attention modules.

    Args:
        model: A HuggingFace PreTrainedModel instance.

    Returns:
        List of (fully-qualified name, module) tuples for attention layers.
    """
    attn_cls_name = _detect_attention_class(model)
    if attn_cls_name is None:
        return []

    results = []
    for name, module in model.named_modules():
        if type(module).__name__ == attn_cls_name:
            results.append((name, module))
    return results


def _compute_kv_dim(model_config: "transformers.PretrainedConfig") -> int:
    """Compute the per-layer KV vector dimension from the model config.

    Handles both standard multi-head attention (MHA) and grouped-query
    attention (GQA) models.

    For GQA models (e.g., Llama-2 70B, Mistral):
        kv_dim = (hidden_size // num_attention_heads) * num_key_value_heads

    For MHA models (e.g., GPT-2, OPT):
        kv_dim = hidden_size

    Args:
        model_config: The HuggingFace model configuration object.

    Returns:
        Integer KV dimension for a single layer.
    """
    hidden_size: int = model_config.hidden_size
    num_attention_heads: int = model_config.num_attention_heads
    head_dim = hidden_size // num_attention_heads

    # GQA: num_key_value_heads < num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_attention_heads)
    kv_dim = head_dim * num_kv_heads

    logger.info(
        "KV dim: %d (hidden=%d, heads=%d, kv_heads=%d, head_dim=%d)",
        kv_dim, hidden_size, num_attention_heads, num_kv_heads, head_dim,
    )
    return kv_dim


def _get_num_layers(model_config: "transformers.PretrainedConfig") -> int:
    """Extract the number of transformer layers from the model config.

    Different architectures use different attribute names for this.

    Args:
        model_config: The HuggingFace model configuration object.

    Returns:
        Number of transformer layers.
    """
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if hasattr(model_config, attr):
            return getattr(model_config, attr)
    raise ValueError(
        "Cannot determine number of layers from model config. "
        f"Available attributes: {list(vars(model_config).keys())}"
    )


# ---------------------------------------------------------------------------
# Main hook class
# ---------------------------------------------------------------------------


class FastKVModelHook:
    """Hooks into a HuggingFace causal LM to route KV cache through Fast-KV.

    Loads a HuggingFace model and tokenizer, detects the model architecture,
    and provides generate/generate_baseline methods that run autoregressive
    generation with and without Fast-KV compression.

    Args:
        model_name: HuggingFace model identifier or local path
            (e.g., "meta-llama/Llama-2-7b-hf", "microsoft/phi-2").
        fast_kv_config: FastKVConfig instance. If None, uses default config.
        device: Device to run the model on. Defaults to "cuda" if available,
            otherwise "cpu".
        torch_dtype: Torch dtype for model loading. Defaults to float16 on
            CUDA, float32 on CPU.
        model_kwargs: Extra keyword arguments forwarded to
            AutoModelForCausalLM.from_pretrained().

    Attributes:
        model: The loaded HuggingFace model.
        tokenizer: The loaded tokenizer.
        fast_kv_cache: The FastKVCache instance managing tier assignment.
        attention_class_name: Detected attention module class name.
    """

    def __init__(
        self,
        model_name: str,
        fast_kv_config: Optional[FastKVConfig] = None,
        device: Optional[str] = None,
        torch_dtype: Optional["torch.dtype"] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.fast_kv_config = fast_kv_config or FastKVConfig()

        # Resolve device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve dtype
        if torch_dtype is None:
            self.torch_dtype = (
                torch.float16 if self.device == "cuda" else torch.float32
            )
        else:
            self.torch_dtype = torch_dtype

        logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, self.device, self.torch_dtype)

        # Load model and tokenizer
        _model_kwargs: Dict[str, Any] = model_kwargs or {}
        _model_kwargs.setdefault("torch_dtype", self.torch_dtype)
        _model_kwargs.setdefault("device_map", self.device)
        # Use eager attention so output_attentions=True works (SDPA doesn't support it)
        _model_kwargs.setdefault("attn_implementation", "eager")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **_model_kwargs
            )
        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_name, e)
            raise

        # Ensure pad token is set (many models lack one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

        # Detect architecture
        self.attention_class_name: Optional[str] = _detect_attention_class(self.model)
        self._attention_modules = _get_attention_modules(self.model)
        logger.info(
            "Found %d attention modules of type '%s'",
            len(self._attention_modules),
            self.attention_class_name,
        )

        # Extract model dimensions
        mc = self.model.config
        self._n_layers = _get_num_layers(mc)
        self._kv_dim = _compute_kv_dim(mc)
        self._num_attention_heads = mc.num_attention_heads
        self._head_dim = mc.hidden_size // mc.num_attention_heads
        self._num_kv_heads = getattr(mc, "num_key_value_heads", mc.num_attention_heads)

        # Initialize FastKVCache
        model_config_dict: Dict[str, Any] = {
            "n_layers": self._n_layers,
            "kv_dim": self._kv_dim,
            "dtype": "float32",
        }
        self.fast_kv_cache = FastKVCache(self.fast_kv_config, model_config_dict)

        # Tracking state
        self._generation_step: int = 0
        self._captured_attentions: Dict[int, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        # Per-step attention history for analysis (layer 0 only)
        self.last_attention_weights: Dict[int, List[Dict[int, float]]] = {0: []}
        self.generation_times: Dict[str, float] = {}
        self.ram_measurements: Dict[str, float] = {}

        # Register attention weight capture hooks
        self._register_attention_hooks()

        logger.info(
            "FastKVModelHook initialized: %d layers, kv_dim=%d, "
            "%d attention hooks registered",
            self._n_layers, self._kv_dim, len(self._hook_handles),
        )

    # ------------------------------------------------------------------
    # Attention weight capture
    # ------------------------------------------------------------------

    def _register_attention_hooks(self) -> None:
        """Register forward hooks on attention modules to capture attention weights.

        The hooks intercept the attention module's output and extract the
        attention weight tensor, storing it keyed by layer index.
        """
        self._remove_attention_hooks()

        for layer_idx, (name, module) in enumerate(self._attention_modules):

            def _make_hook(idx: int):
                """Create a closure that captures layer_idx."""

                def _hook(
                    mod: torch.nn.Module,
                    input: Any,
                    output: Any,
                ) -> None:
                    # output_attentions=True makes the module return
                    # (attn_output, attn_weights, ...) instead of just
                    # (attn_output, None, ...)
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            self._captured_attentions[idx] = attn_weights.detach()

                return _hook

            handle = module.register_forward_hook(_make_hook(layer_idx))
            self._hook_handles.append(handle)

        logger.debug("Registered %d attention forward hooks", len(self._hook_handles))

    def _remove_attention_hooks(self) -> None:
        """Remove all registered attention capture hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _extract_attention_weights_for_ise(
        self, layer_idx: int, seq_len: int, token_ids: List[int]
    ) -> Dict[int, float]:
        """Convert captured attention weight tensor to ISE-compatible dict.

        Takes the attention weights from the last query position (the new
        token attending to all past tokens) and averages across heads to
        produce a single importance signal per token position.

        Args:
            layer_idx: Layer index to extract from.
            seq_len: Current sequence length.
            token_ids: Token position IDs corresponding to sequence positions.

        Returns:
            Dict mapping token_id -> average attention weight (float).
        """
        attn_weights: Dict[int, float] = {}

        if layer_idx not in self._captured_attentions:
            # No attention captured for this layer; return uniform
            if token_ids:
                uniform = 1.0 / len(token_ids)
                return {tid: uniform for tid in token_ids}
            return {}

        attn_tensor = self._captured_attentions[layer_idx]
        # Shape: (batch, num_heads, query_len, key_len)
        # We want the last query position's attention over all keys.
        try:
            last_query_attn = attn_tensor[0, :, -1, :]  # (num_heads, key_len)
            avg_attn = last_query_attn.mean(dim=0).cpu().float().numpy()  # (key_len,)
        except Exception as e:
            logger.debug(
                "Could not extract attention for layer %d: %s", layer_idx, e
            )
            if token_ids:
                uniform = 1.0 / len(token_ids)
                return {tid: uniform for tid in token_ids}
            return {}

        # Map position -> token_id with the attention weight
        n_positions = min(len(avg_attn), len(token_ids))
        for pos in range(n_positions):
            attn_weights[token_ids[pos]] = float(avg_attn[pos])

        return attn_weights

    # ------------------------------------------------------------------
    # KV cache interception
    # ------------------------------------------------------------------

    def _intercept_kv_cache(
        self,
        past_key_values: Tuple,
        token_ids: List[int],
        token_texts: List[str],
        current_step: int,
        new_token_count: int = 1,
    ) -> Tuple:
        """Route the model's KV cache through the Fast-KV system.

        For each layer, extracts the K/V tensors, converts to numpy,
        pushes new tokens through FastKVCache.update(), then reconstructs
        the full past_key_values from FastKVCache.

        Args:
            past_key_values: The model's past_key_values output. A tuple of
                (key, value) tensors per layer, each of shape
                (batch, num_kv_heads, seq_len, head_dim).
            token_ids: Ordered list of token position IDs (0-based).
            token_texts: Corresponding token text strings.
            current_step: Current generation step number.
            new_token_count: Number of new tokens added this step.

        Returns:
            Reconstructed past_key_values tuple with cold tokens replaced
            by decompressed approximations.
        """
        seq_len = len(token_ids)
        new_start = seq_len - new_token_count

        reconstructed_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(self._n_layers):
            # Extract K/V tensors for this layer
            if hasattr(past_key_values, "key_cache"):
                # DynamicCache style
                key_tensor = past_key_values.key_cache[layer_idx]
                value_tensor = past_key_values.value_cache[layer_idx]
            else:
                # Tuple style: (key, value) per layer
                key_tensor = past_key_values[layer_idx][0]
                value_tensor = past_key_values[layer_idx][1]

            # Shape: (batch, num_kv_heads, seq_len, head_dim)
            # Flatten heads into a single dim for FastKV: (seq_len, kv_dim)
            batch_size = key_tensor.shape[0]
            device = key_tensor.device
            dtype = key_tensor.dtype

            # Work with the first batch element
            k = key_tensor[0]  # (num_kv_heads, seq_len, head_dim)
            v = value_tensor[0]  # (num_kv_heads, seq_len, head_dim)

            # Reshape: (seq_len, num_kv_heads * head_dim)
            k_flat = k.permute(1, 0, 2).reshape(seq_len, -1)
            v_flat = v.permute(1, 0, 2).reshape(seq_len, -1)

            # Extract attention weights for ISE update
            attn_weights = self._extract_attention_weights_for_ise(
                layer_idx, seq_len, token_ids
            )

            # Record attention history for layer 0 (for analysis)
            if layer_idx == 0 and attn_weights:
                if 0 not in self.last_attention_weights:
                    self.last_attention_weights[0] = []
                self.last_attention_weights[0].append(dict(attn_weights))

            # Process only the NEW tokens through FastKV update
            for pos in range(new_start, seq_len):
                tid = token_ids[pos]
                token_text = token_texts[pos] if pos < len(token_texts) else ""

                k_np = k_flat[pos].cpu().float().numpy()
                v_np = v_flat[pos].cpu().float().numpy()

                self.fast_kv_cache.update(
                    layer_id=layer_idx,
                    token_id=tid,
                    token_text=token_text,
                    key_vector=k_np,
                    value_vector=v_np,
                    attention_weights=attn_weights,
                    current_step=current_step,
                )

            # Reconstruct full KV from FastKVCache
            keys_np, values_np = self.fast_kv_cache.get_kv_cache(
                layer_idx, token_ids
            )
            # keys_np shape: (seq_len, kv_dim), values_np same

            # Convert back to torch tensors
            k_reconstructed = torch.from_numpy(keys_np).to(dtype=dtype, device=device)
            v_reconstructed = torch.from_numpy(values_np).to(dtype=dtype, device=device)

            # Reshape back: (seq_len, kv_dim) -> (1, num_kv_heads, seq_len, head_dim)
            k_out = (
                k_reconstructed
                .reshape(seq_len, self._num_kv_heads, self._head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
            )
            v_out = (
                v_reconstructed
                .reshape(seq_len, self._num_kv_heads, self._head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
            )

            reconstructed_layers.append((k_out, v_out))

        # Rebuild into the same cache format the model expects
        rebuilt = self._pack_past_key_values(past_key_values, reconstructed_layers)
        return rebuilt

    def _pack_past_key_values(
        self,
        original_past: Any,
        layers: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Any:
        """Pack reconstructed layer tensors into the format the model expects.

        Always returns a DynamicCache object, as modern transformers models
        expect cache objects with methods like get_seq_length().

        Args:
            original_past: The original past_key_values object (for format detection).
            layers: List of (key_tensor, value_tensor) per layer.

        Returns:
            Past key values as a DynamicCache.
        """
        try:
            from transformers import DynamicCache
        except ImportError:
            DynamicCache = None

        if hasattr(original_past, "key_cache") and hasattr(original_past, "value_cache"):
            # DynamicCache: replace in-place
            for layer_idx, (k, v) in enumerate(layers):
                original_past.key_cache[layer_idx] = k
                original_past.value_cache[layer_idx] = v
            return original_past
        elif DynamicCache is not None:
            # Convert tuple-of-tuples to DynamicCache for compatibility
            cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(layers):
                cache.update(k, v, layer_idx)
            return cache
        else:
            # Last resort: tuple-of-tuples
            return tuple(layers)

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        output_attentions: bool = True,
    ) -> str:
        """Generate text with Fast-KV cache compression active.

        Runs a token-by-token autoregressive generation loop, intercepting
        the KV cache at each step and routing it through FastKVCache.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Only used if do_sample=True.
            do_sample: Whether to sample or use greedy decoding.
            output_attentions: Whether to request attention weights from
                the model for ISE updates. Disabling may improve speed
                but reduces ISE accuracy (falls back to uniform attention).

        Returns:
            The generated text including the prompt.
        """
        logger.info(
            "Generating with Fast-KV (max_new_tokens=%d, output_attentions=%s)",
            max_new_tokens, output_attentions,
        )

        # Reset cache for a fresh generation
        self.reset()

        # Measure initial memory
        ram_before = self._measure_ram_mb()
        t_start = time.perf_counter()

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generated_ids = input_ids.clone()
        past_key_values = None

        # Build initial token ID list and token texts
        prompt_token_ids_list: List[int] = input_ids[0].tolist()
        all_token_ids: List[int] = list(range(len(prompt_token_ids_list)))
        all_token_texts: List[str] = [
            self.tokenizer.decode([tid], skip_special_tokens=False)
            for tid in prompt_token_ids_list
        ]

        for step in range(max_new_tokens):
            self._generation_step = step
            self._captured_attentions.clear()

            with torch.no_grad():
                model_kwargs: Dict[str, Any] = {
                    "use_cache": True,
                    "output_attentions": output_attentions,
                }

                if past_key_values is None:
                    # First pass: full prompt
                    outputs = self.model(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        **model_kwargs,
                    )
                    new_token_count = len(prompt_token_ids_list)
                else:
                    # Subsequent passes: only the new token
                    outputs = self.model(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        **model_kwargs,
                    )
                    new_token_count = 1

            # Get next token via greedy or sampling
            logits = outputs.logits[:, -1, :]
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = logits.argmax(dim=-1, keepdim=True)

            # Append the new token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1), dtype=attention_mask.dtype, device=self.device
                        ),
                    ],
                    dim=-1,
                )

            # Intercept and route KV cache through Fast-KV
            # NOTE: the model's past_key_values contains KV for all tokens
            # that were in input_ids this step. The new token picked from
            # logits does NOT have its KV in the cache yet — it will be
            # processed by the model in the next step.
            raw_past = outputs.past_key_values
            try:
                past_key_values = self._intercept_kv_cache(
                    past_key_values=raw_past,
                    token_ids=all_token_ids,
                    token_texts=all_token_texts,
                    current_step=step,
                    new_token_count=new_token_count,
                )
            except Exception as e:
                logger.error(
                    "Fast-KV interception failed at step %d: %s. "
                    "Falling back to uncompressed cache.",
                    step, e,
                )
                past_key_values = raw_past

            # Track the new token AFTER interception (its KV will be
            # added to the cache when the model processes it next step)
            new_tid = next_token_id.item()
            new_position = len(all_token_ids)
            all_token_ids.append(new_position)
            all_token_texts.append(
                self.tokenizer.decode([new_tid], skip_special_tokens=False)
            )

            # Check for EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                logger.debug("EOS token generated at step %d", step)
                break

        t_elapsed = time.perf_counter() - t_start
        ram_after = self._measure_ram_mb()
        tokens_generated = generated_ids.shape[1] - input_ids.shape[1]

        logger.info(
            "Generation complete: %d tokens in %.2fs (%.1f tok/s). "
            "RAM delta: %.1f MB",
            tokens_generated,
            t_elapsed,
            tokens_generated / t_elapsed if t_elapsed > 0 else 0,
            ram_after - ram_before,
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """Generate text WITHOUT Fast-KV compression for comparison.

        Uses the standard model.generate() method with default KV caching.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Only used if do_sample=True.
            do_sample: Whether to sample or use greedy decoding.

        Returns:
            The generated text including the prompt.
        """
        logger.info("Generating baseline (no Fast-KV, max_new_tokens=%d)", max_new_tokens)

        ram_before = self._measure_ram_mb()
        t_start = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        t_elapsed = time.perf_counter() - t_start
        ram_after = self._measure_ram_mb()
        tokens_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]

        logger.info(
            "Baseline complete: %d tokens in %.2fs (%.1f tok/s). "
            "RAM delta: %.1f MB",
            tokens_generated,
            t_elapsed,
            tokens_generated / t_elapsed if t_elapsed > 0 else 0,
            ram_after - ram_before,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Reporting and lifecycle
    # ------------------------------------------------------------------

    def get_memory_report(self) -> str:
        """Return a formatted memory usage report from FastKVCache.

        Delegates to FastKVCache.get_memory_report() which aggregates
        tier distribution, RAM usage, and promotion/demotion statistics
        across all layers.

        Returns:
            Multi-line formatted string with the memory report.
        """
        return self.fast_kv_cache.get_memory_report()

    def reset(self) -> None:
        """Clear all Fast-KV state for a new conversation.

        Resets the FastKVCache (including ISE and all tier managers),
        clears captured attention weights, and resets the step counter.
        """
        self.fast_kv_cache.reset()
        self._captured_attentions.clear()
        self._generation_step = 0
        self.last_attention_weights = {0: []}
        self.generation_times = {}
        self.ram_measurements = {}
        logger.info("FastKVModelHook reset")

    def cleanup(self) -> None:
        """Remove all hooks and free resources.

        Should be called when the hook is no longer needed to prevent
        memory leaks from lingering forward hooks.
        """
        self._remove_attention_hooks()
        self._captured_attentions.clear()
        logger.info("FastKVModelHook cleaned up")

    def __del__(self) -> None:
        """Ensure hooks are removed on garbage collection."""
        try:
            self._remove_attention_hooks()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_ram_mb() -> float:
        """Measure current process RSS memory usage in MB.

        Returns:
            RSS memory in megabytes, or 0.0 if psutil is not available.
        """
        if psutil is None:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary of the loaded model's configuration.

        Returns:
            Dictionary with model name, architecture details, and
            Fast-KV configuration.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "n_layers": self._n_layers,
            "kv_dim": self._kv_dim,
            "num_attention_heads": self._num_attention_heads,
            "num_kv_heads": self._num_kv_heads,
            "head_dim": self._head_dim,
            "attention_class": self.attention_class_name,
            "fast_kv_config": {
                "hot_threshold": self.fast_kv_config.hot_threshold,
                "cold_threshold": self.fast_kv_config.cold_threshold,
                "bits_subtier_2a": self.fast_kv_config.bits_subtier_2a,
                "bits_subtier_2b": self.fast_kv_config.bits_subtier_2b,
                "bits_subtier_2c": self.fast_kv_config.bits_subtier_2c,
            },
        }

    def __repr__(self) -> str:
        return (
            f"FastKVModelHook(model={self.model_name!r}, "
            f"device={self.device!r}, "
            f"layers={self._n_layers}, "
            f"kv_dim={self._kv_dim}, "
            f"attn_class={self.attention_class_name!r})"
        )
