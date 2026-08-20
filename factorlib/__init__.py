"""factorlib -- a small, opinionated harness for cross-sectional factor research.

Public extract of a private crypto factor research lab. The library is the
methodology; the proprietary signals are not included.

    from factorlib import evaluate, portfolio, universe, factors, tearsheet
"""

from . import data, evaluate, factors, portfolio, tearsheet, universe

__all__ = ["data", "evaluate", "factors", "portfolio", "tearsheet", "universe"]
__version__ = "0.1.0"
