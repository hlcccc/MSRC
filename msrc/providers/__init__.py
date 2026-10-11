"""Model adapters.

Importing this package must stay free when the optional extras are absent, so
nothing here imports torch at module level. The decision layer never imports it
at all.

    from msrc.providers import HFLLaVAProvider      # needs the [hf] extra
    from msrc.providers import HFQwenVLProvider     # needs the [hf] extra
    from msrc.providers import HFInternVLProvider   # needs the [hf] extra

All are the same class underneath; they differ only in which transformers model
class they load. The InternVL one needs the ``-hf`` conversion of the checkpoint,
not the original release: see its docstring for why.
"""

from msrc.providers.hf_vision import (
    HFInternVLProvider,
    HFLLaVAProvider,
    HFOCRProvider,
    HFQwenVLProvider,
    HFVisionLanguageProvider,
)

__all__ = [
    "HFInternVLProvider",
    "HFLLaVAProvider",
    "HFOCRProvider",
    "HFQwenVLProvider",
    "HFVisionLanguageProvider",
]
