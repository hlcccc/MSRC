"""Model adapters.

Importing this package must stay free when the optional extras are absent, so
nothing here imports torch at module level. The decision layer never imports it
at all.

    from msrc.providers import HFLLaVAProvider      # needs the [hf] extra
    from msrc.providers import HFQwenVLProvider     # needs the [hf] extra

Both are the same class underneath; they differ only in which transformers model
class they load.
"""

from msrc.providers.hf_vision import (
    HFLLaVAProvider,
    HFOCRProvider,
    HFQwenVLProvider,
    HFVisionLanguageProvider,
)

__all__ = [
    "HFLLaVAProvider",
    "HFOCRProvider",
    "HFQwenVLProvider",
    "HFVisionLanguageProvider",
]
