"""
Gaunt utilities package.

Modules:
- helpers.gaunt_cache_wigxjpf: cache helpers for Gaunt tensors
- gaunt_vectorized / gaunt_vectorized_wigxjpf: vectorized coefficient builders
- assemble_gaunt_checkpoints: merge per-L checkpoint shards
"""

try:
    from gaunt.helpers import gaunt_cache_wigxjpf  # re-export for convenience
    _HAS_WIGXJPF = True
except ImportError:
    gaunt_cache_wigxjpf = None
    _HAS_WIGXJPF = False

__all__ = [
    "gaunt_vectorized",
    "gaunt_vectorized_wigxjpf",
    "assemble_gaunt_checkpoints",
]

if _HAS_WIGXJPF:
    __all__.append("gaunt_cache_wigxjpf")
