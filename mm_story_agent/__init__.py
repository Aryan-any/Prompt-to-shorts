import sys

from .utils.import_utils import _LazyModule
from .base import TOOL_REGISTRY, register_tool, init_tool_instance
# from .modality_agents.llm import GeminiAgent, OpenAIAgent
from .modality_agents.llm import  OpenAIAgent
from .modality_agents.story_agent import QAOutlineStoryWriter
from .modality_agents.music_agent import MusicGenAgent
from .modality_agents.sound_agent import AudioLDM2Agent
from .modality_agents.speech_agent import GoogleTTSAgent
from .modality_agents.image_agent import StoryDiffusionAgent
from .modality_agents.freesound_agent import FreesoundSfxAgent, FreesoundMusicAgent
from .mm_story_agent import MMStoryAgent


_import_structure = {
    'modality_agents': [
        'QAOutlineStoryWriter',
        'MusicGenAgent',
        'AudioLDM2Agent',
        'CosyVoiceAgent',
        'OpenAIAgent',
        'StoryDiffusionAgent',
        'GeminiAgent',
        'QwenAgent',
        'FreesoundSfxAgent',
        'FreesoundMusicAgent'
    ],
    'mm_story_agent': [
        'MMStoryAgent'
    ],
    'video_compose_agent': [
        'SlideshowVideoComposeAgent'
    ],
    'base': [
        'init_tool_instance'
    ],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)

__all__ = [
    'TOOL_REGISTRY',
    'register_tool',
    'init_tool_instance',
    'MMStoryAgent',
    'GeminiAgent',
    'OpenAIAgent',
    'QAOutlineStoryWriter',
    'MusicGenAgent',
    'AudioLDM2Agent',
    'GoogleTTSAgent',
    'StoryDiffusionAgent',
    'FreesoundSfxAgent',
    'FreesoundMusicAgent'
]