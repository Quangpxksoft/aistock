# utils/batch_utils.py – generic batch processing helper

from __future__ import annotations

import gc
from typing import Callable, Iterable

try:
    import tensorflow as tf
except ImportError:
    tf = None  # noqa: N816 – ok if TF not installed


def process_batch(
    tickers: Iterable[str],
    start_date,
    end_date,
    *,
    step_fn: Callable[[object, str], None],
    load_fn: Callable[[str, object, object], object] | None = None,
):
    """Process a batch of *tickers*.

    Args:
        tickers: Iterable of stock symbols.
        start_date, end_date: passed to *load_fn*.
        step_fn: user callback **step_fn(df, ticker)** called per ticker.
        load_fn: optional custom loader. If None, `step_fn` must ignore df.
    """
    load_fn = load_fn or (lambda t, s, e: None)
    for t in tickers:
        df = load_fn(t, start_date, end_date)
        step_fn(df, t)
    # ---- Free memory -------------------------------------------------------
    gc.collect()
    if tf is not None:
        tf.keras.backend.clear_session()
