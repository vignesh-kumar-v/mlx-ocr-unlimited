"""KV caches for autoregressive decoding.

``PrefillWindowCache`` implements DeepSeek-OCR's decode-time attention: every
prompt/image (prefill) token is kept permanently, plus a rotating window of the
most recent ``window`` generated tokens. This matches the reference
``SlidingWindowLlamaAttention`` ring buffer — the image context is never
evicted — while bounding the cache at ``prefill_len + window`` instead of
letting it grow with the full generation.
"""

import mlx.core as mx


class PrefillWindowCache:
    def __init__(self, window: int):
        self.window = window
        self.offset = 0  # absolute position of the next token (for RoPE)
        self._pk = self._pv = None  # fixed prefill keys/values
        self._dk = self._dv = None  # rotating decode window

    def update_and_fetch(self, keys, values):
        s = keys.shape[2]
        if self._pk is None:
            # Prefill: keep everything and attend over the full prompt.
            self._pk, self._pv = keys, values
            self.offset = s
            return keys, values

        self.offset += s
        if self._dk is None:
            self._dk, self._dv = keys, values
        else:
            self._dk = mx.concatenate([self._dk, keys], axis=2)
            self._dv = mx.concatenate([self._dv, values], axis=2)
        if self._dk.shape[2] > self.window:
            self._dk = self._dk[:, :, -self.window :, :]
            self._dv = self._dv[:, :, -self.window :, :]

        keys = mx.concatenate([self._pk, self._dk], axis=2)
        values = mx.concatenate([self._pv, self._dv], axis=2)
        return keys, values
