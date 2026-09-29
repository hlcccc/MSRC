"""Model adapters.

Importing this package must stay free when the optional extras are absent, so
nothing here imports torch at module level. The decision layer never imports it
at all.

    from msrc.providers import HFLLaVAProvider      # needs the [hf] extra
"""

from msrc.providers.hf_llava import HFLLaVAProvider, HFOCRProvider

__all__ = ["HFLLaVAProvider", "HFOCRProvider"]
