from abc import ABC
from typing import Dict, Type
import importlib

class ToolRegistry(dict):
    def __init__(self):
        super().__init__()
        print("Initializing ToolRegistry")

    def __setitem__(self, key, value):
        print(f"Registering tool: {key}")
        super().__setitem__(key, value)

    def __getitem__(self, key):
        print(f"Accessing tool: {key}")
        return super().__getitem__(key)

TOOL_REGISTRY = ToolRegistry()

def register_tool(name: str):
    def wrapper(cls: Type):
        print(f"Registering {name}: {cls.__name__}")
        TOOL_REGISTRY[name] = cls
        return cls
    return wrapper

# Import all tools at startup
def import_all_tools():
    print("Importing all tools...")
    # from .modality_agents.llm import GeminiAgent, OpenAIAgent
    from .modality_agents.llm import  OpenAIAgent
    from .modality_agents.story_agent import QAOutlineStoryWriter
    from .modality_agents.music_agent import MusicGenAgent
    from .modality_agents.sound_agent import AudioLDM2Agent
    from .modality_agents.speech_agent import GoogleTTSAgent, ElevenLabsAgent
    from .modality_agents.image_agent import StoryDiffusionAgent
    from .modality_agents.freesound_agent import FreesoundSfxAgent, FreesoundMusicAgent

def init_tool_instance(cfg: Dict):
    if not TOOL_REGISTRY:
        import_all_tools()
    import_all_tools()
    print(f"Initializing tool: {cfg['tool']}")
    if cfg["tool"] not in TOOL_REGISTRY:
        raise KeyError(f"Tool {cfg['tool']} not found in registry. Available tools: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[cfg["tool"]](cfg["cfg"])

# Import tools when module is loaded
import_all_tools()