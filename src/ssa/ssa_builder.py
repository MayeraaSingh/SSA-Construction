"""Lazy SSA construction using sealed blocks.

Implements the core SSA construction algorithm that defers phi-function
insertion until blocks are sealed, as described in the paper:

    "Simple and Efficient Construction of Static Single Assignment Form"
"""
