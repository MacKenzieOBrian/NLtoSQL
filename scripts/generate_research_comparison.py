#!/usr/bin/env python3
"""All-in-one wrapper: collect raw tables, then format the analysis outputs."""

from nl2sql.evaluation.research_comparison import generate, main

__all__ = ["generate", "main"]


if __name__ == "__main__":
    main()
