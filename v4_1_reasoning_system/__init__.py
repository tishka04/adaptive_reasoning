"""
v4_1_reasoning_system — Adaptive Reasoning Architecture

Core loop:
    Problem → LLM parse → latent state → candidate reasoning actions
    → JEPA-style latent consequence prediction → EBM routing
    → specialized solver/tool execution → external verification
    → repair or memory update → repeat.
"""

__version__ = "0.1.0"
