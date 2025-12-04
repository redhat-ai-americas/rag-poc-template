# NumPy 2.x compatibility shim for libraries expecting np.float_
try:
    import numpy as np  # type: ignore

    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
except Exception:
    pass

from .wiki_processor import WikiProcessor

__all__ = ["WikiProcessor"]
