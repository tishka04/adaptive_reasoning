from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm

# Conditional imports for agents with optional dependencies
# NOTE: each optional agent is wrapped in a broad `except Exception` (not
# just ImportError) because third-party API changes (e.g. Pillow renaming
# private symbols used by langgraph_thinking) raise AttributeError at
# import time, which would otherwise take down the whole package and
# block the unrelated AdaptiveReasoning agent from loading.
try:
    from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
except Exception:
    LangGraphFunc = None  # type: ignore
    LangGraphTextOnly = None  # type: ignore

try:
    from .templates.langgraph_random_agent import LangGraphRandom
except Exception:
    LangGraphRandom = None  # type: ignore

try:
    from .templates.langgraph_thinking import LangGraphThinking
except Exception:
    LangGraphThinking = None  # type: ignore

try:
    from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
except Exception:
    LLM = None  # type: ignore
    FastLLM = None  # type: ignore
    GuidedLLM = None  # type: ignore
    ReasoningLLM = None  # type: ignore

try:
    from .templates.multimodal import MultiModalLLM
except Exception:
    MultiModalLLM = None  # type: ignore

try:
    from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
except Exception:
    SmolCodingAgent = None  # type: ignore
    SmolVisionAgent = None  # type: ignore

from .templates.random_agent import Random
try:
    from .templates.reasoning_agent import ReasoningAgent
except Exception:
    ReasoningAgent = None  # type: ignore
from .templates.adaptive_reasoning_agent import AdaptiveReasoning

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
if ReasoningAgent is not None:
    AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent
AVAILABLE_AGENTS["adaptivereasoning"] = AdaptiveReasoning

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "AdaptiveReasoning",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
]
