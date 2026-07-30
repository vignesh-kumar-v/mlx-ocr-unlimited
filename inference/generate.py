"""Autoregressive generation for DeepSeek-OCR."""

import time

import mlx.core as mx


class SlidingWindowNoRepeatNgramProcessor:
    """Ban tokens that would repeat an n-gram seen within a sliding window
    (aligned with the reference SGLang DeepseekOCRNoRepeatNGramLogitProcessor)."""

    def __init__(self, ngram_size, window, whitelist_token_ids=None):
        self.ngram_size = ngram_size
        self.window = window
        self.whitelist = set(whitelist_token_ids) if whitelist_token_ids else set()

    def get_banned(self, sequence):
        if len(sequence) < self.ngram_size:
            return set()
        search_start = max(0, len(sequence) - self.window)
        search_end = len(sequence) - self.ngram_size + 1
        if search_end <= search_start:
            return set()
        prefix = (
            tuple(sequence[-(self.ngram_size - 1) :]) if self.ngram_size > 1 else ()
        )
        banned = set()
        for idx in range(search_start, search_end):
            ngram = sequence[idx : idx + self.ngram_size]
            if self.ngram_size == 1 or tuple(ngram[:-1]) == prefix:
                banned.add(ngram[-1])
        banned.difference_update(self.whitelist)
        return banned


def _sample(logits, temperature):
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits * (1 / temperature))


def generate(
    model,
    input_ids,
    images=None,
    images_seq_mask=None,
    images_spatial_crop=None,
    max_tokens=8192,
    temperature=0.0,
    eos_token_id=1,
    no_repeat_ngram_size=0,
    ngram_window=0,
    stream=None,
    verbose=False,
):
    """Greedy/temperature decoding. Returns (generated_token_ids, tokens_per_sec)."""
    cache = model.make_cache()
    tokens = list(input_ids[0].tolist())
    generated = []

    ngram = None
    if no_repeat_ngram_size > 0 and ngram_window > 0:
        ngram = SlidingWindowNoRepeatNgramProcessor(no_repeat_ngram_size, ngram_window)

    # Prefill (runs the vision encoders and fills the cache).
    logits = model(
        input_ids=input_ids,
        cache=cache,
        images=images,
        images_seq_mask=images_seq_mask,
        images_spatial_crop=images_spatial_crop,
    )[0, -1]

    start = None
    for _ in range(max_tokens):
        if ngram is not None:
            banned = ngram.get_banned(tokens)
            if banned:
                penalty = mx.zeros(logits.shape).at[mx.array(list(banned))].add(-1e9)
                logits = logits + penalty

        token = _sample(logits, temperature)
        mx.async_eval(token)
        tok_id = token.item()
        if start is None:  # start timing after the first (prefill-bound) token
            start = time.time()

        tokens.append(tok_id)
        generated.append(tok_id)
        if stream is not None:
            stream(tok_id)
        if tok_id == eos_token_id:
            break

        logits = model(input_ids=mx.array([[tok_id]]), cache=cache)[0, -1]

    tps = (
        (len(generated) - 1) / (time.time() - start)
        if start and len(generated) > 1
        else 0.0
    )
    if verbose:
        print(f"\n[decode] {len(generated)} tokens, {tps:.1f} tok/s")
    return generated, tps
