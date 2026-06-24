#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import types
from contextlib import contextmanager
from functools import partial
from typing import Callable, Generator, Optional

from torch import nn


def register_attention_forward_hook(
    model: nn.Module,
    hook: Callable,
    attention_wrapper: Optional[Callable],
) -> Generator:
    """Registers an hook to each attention layer in the model.

    Optionally, it substitutes the attention_call in the layer with a wrapper `attention_wrapper`,
    it is expected to have the following signature:
        wrapped_attention_call(
            q, k, v, mask, dropout_p, is_causal, original_attention_call, attn_module
        )
    It is useful to store in the attn_module itself information that can later be accessed by the hook.

    Args:
        model (nn.Module): the transformer model to patch
        hook (Callable): the hook to register in the forward of each attention
        attention_wrapper (Optional[Callable]): the attention wrapper to use

    Returns:
        a context manager to activate the hooks
    """

    @contextmanager
    def _register_attention_forward_hook(
        model: nn.Module,
        hook: Callable,
    ) -> Generator:
        hooks = []
        original_attentions_fn = []
        try:
            for idx, layer in enumerate(model.layers):
                if attention_wrapper is not None:
                    original_attentions_fn.append(layer.attn._attention_call)
                    layer.attn._attention_call = partial(
                        attention_wrapper,
                        original_attention_call=layer.attn._attention_call,
                        attn_module=layer.attn,
                    )
                hooks.append(
                    layer.attn.register_forward_hook(
                        partial(hook, layer_idx=idx), with_kwargs=True
                    )
                )
            yield

        finally:
            if attention_wrapper is not None:
                for layer, attn_call in zip(model.layers, original_attentions_fn):
                    layer.attn._attention_call = attn_call

            for forward_hook in hooks:
                forward_hook.remove()

    return _register_attention_forward_hook(model=model, hook=hook)


def monkeypatch_multihead_attention(
    model: nn.Module,
    forward_post_hook: Callable,
    monkeypatch_mh_attention_forward: Callable,
    monkeypatch_mh_attention_setup_cache: Callable,
) -> Generator:
    """Monkeypatches MultiHeadAttention (MHA) layers within a given model.

    This function temporarily modifies the behavior of MHA layers by:
    1. Replacing the `forward` method of each MHA layer with the
       provided `monkeypatch_mh_attention_forward` callable.
    2. Replacing the `setup_cache` method of each MHA layer with the
       provided `monkeypatch_mh_attention_setup_cache` callable.
    3. Registering `forward_post_hook` as a forward hook on each MHA layer.
       This hook is executed *after* the (monkeypatched) `forward` method.

    The modifications are applied when the returned generator is used as a
    context manager (e.g., in a `with` statement). Upon exiting the context,
    the original MHA methods are restored, and the registered hooks are removed.

    This is useful for observing or modifying the MHA layer's behavior,
    potentially storing information within the `attn_module` (accessible via
    `self` in the patched methods or `module` in the hook) for later access.

    Args:
        model (nn.Module): The transformer model whose MultiHeadAttention
            layers will be patched.
        forward_post_hook (Callable): A callable to be registered as a
            forward hook on each MHA layer. It's executed after the
            (monkeypatched) `forward` method. It will be called with
            `(module, input, output, **kwargs)` where `kwargs` will include
            `layer_idx` (the index of the current layer).
        monkeypatch_mh_attention_forward (Callable): A callable that will
            replace the `forward` method of each MHA layer. It should have a
            signature compatible with the original MHA's `forward` method
            (e.g., `def new_forward(self, x, ...):`).
        monkeypatch_mh_attention_setup_cache (Callable): A callable that will
            replace the `setup_cache` method of each MHA layer. It should
            have a signature compatible with the original MHA's `setup_cache`
            method (e.g., `def new_setup_cache(self, batch_size, max_seq_len, dtype, device):`).

    Returns:
        Generator: A generator that, when used as a context manager,
            applies the monkeypatches and registers the hooks. Upon exiting
            the context, it restores the original MHA methods and removes the
            hooks.
    """

    @contextmanager
    def _monkeypatch_multihead_attention(
        model: nn.Module,
        forward_post_hook: Callable,
    ) -> Generator:
        hooks = []
        original_mh_attentions_fn = []
        original_mh_setup_cache = []
        try:
            for idx, layer in enumerate(model.layers):
                original_mh_attentions_fn.append(layer.attn.forward)
                original_mh_setup_cache.append(layer.attn.setup_cache)
                layer.attn._layer_idx = idx
                layer.attn.forward = types.MethodType(
                    monkeypatch_mh_attention_forward, layer.attn
                )
                layer.attn.setup_cache = types.MethodType(
                    monkeypatch_mh_attention_setup_cache, layer.attn
                )
                hooks.append(
                    layer.attn.register_forward_hook(
                        partial(forward_post_hook, layer_idx=idx), with_kwargs=True
                    )
                )
            yield

        finally:
            for layer, mh_attn_call, mh_setup_cache in zip(
                model.layers, original_mh_attentions_fn, original_mh_setup_cache
            ):
                layer.attn.forward = mh_attn_call
                layer.attn.setup_cache = mh_setup_cache

            for forward_hook in hooks:
                forward_hook.remove()

    return _monkeypatch_multihead_attention(
        model=model, forward_post_hook=forward_post_hook
    )
