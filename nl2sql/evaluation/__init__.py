"""Official evaluation surface for the simplified dissertation workflow.

The active analysis path is deliberately small: load the manual final pack,
build a condition summary, and run the fixed EX-only pairwise tests.
"""

from .final_pack import build_tables_from_pack
from .simple_stats import build_pairwise_tests, build_summary_by_condition

__all__ = [
    "build_pairwise_tests",
    "build_summary_by_condition",
    "build_tables_from_pack",
]
