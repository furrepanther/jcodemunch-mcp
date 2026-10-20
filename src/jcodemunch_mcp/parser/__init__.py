"""Parser package for extracting symbols from source code."""

from .symbols import Symbol, make_symbol_id, compute_content_hash
from .languages import LanguageSpec, LANGUAGE_REGISTRY, LANGUAGE_EXTENSIONS, PYTHON_SPEC, get_language_for_path
from .hierarchy import SymbolNode, build_symbol_tree, flatten_tree


def parse_file(*args, **kwargs):
    """Lazily import the heavy tree-sitter extractor on first use."""
    from .extractor import parse_file as _parse_file

    return _parse_file(*args, **kwargs)

__all__ = [
    "Symbol",
    "make_symbol_id",
    "compute_content_hash",
    "LanguageSpec",
    "LANGUAGE_REGISTRY",
    "LANGUAGE_EXTENSIONS",
    "PYTHON_SPEC",
    "get_language_for_path",
    "parse_file",
    "SymbolNode",
    "build_symbol_tree",
    "flatten_tree",
]
