"""Format adapters. Import :func:`sonar_core.parsers.base.load` to auto-dispatch."""

from sonar_core.parsers.base import NAV_DTYPE, ParserError, PingArray, SonarParser, load

__all__ = ["NAV_DTYPE", "ParserError", "PingArray", "SonarParser", "load"]
