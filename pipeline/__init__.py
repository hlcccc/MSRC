"""Acceptance pipeline: the five entry points behind scripts/01..05.

The decision layer (``msrc``) and the benchmark adapters (``evaluation``) stay
importable on a CPU with no weights. Everything that needs a GPU, a checkpoint
or a delivered asset lives here, so ``pip install -e .`` without the ``hf``
extra still gives a working decision layer.
"""
