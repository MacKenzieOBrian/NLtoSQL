#!/usr/bin/env python3
"""CLI wrapper for the research comparison generator."""

from nl2sql.evaluation.research_comparison import generate, main

__all__ = ["generate", "main"]


if __name__ == "__main__":
    main()
