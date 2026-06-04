import sys

from ..utils.import_utils import _LazyModule
from .story_agent import QAOutlineStoryWriter
# from .llm import GeminiAgent, OpenAIAgent
from .llm import  OpenAIAgent
from .music_agent import MusicGenAgent
from .sound_agent import AudioLDM2Agent
from .speech_agent import GoogleTTSAgent
from .image_agent import StoryDiffusionAgent
from .freesound_agent import FreesoundSfxAgent, FreesoundMusicAgent
from .video_compose_agent import SlideshowVideoComposeAgent

_import_structure = {
    'story_agent': [
        'QAOutlineStoryWriter',
    ],
    'music_agent': [
        'MusicGenAgent'
    ],
    'sound_agent': [
        'AudioLDM2Agent'
    ],
    'speech_agent': [
        'GoogleTTSAgent'
    ],
    'image_agent': [
        'StoryDiffusionAgent'
    ],
    'llm': [
        'GeminiAgent',
        'OpenAIAgent'
    ],
    "freesound_agent": [
        "FreesoundSfxAgent",
        "FreesoundMusicAgent"
    ],
    'video_compose_agent': [
        'SlideshowVideoComposeAgent'
    ]
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)

__all__ = [
    'QAOutlineStoryWriter',
    'GeminiAgent',
    'OpenAIAgent',
    'MusicGenAgent',
    'AudioLDM2Agent',
    'GoogleTTSAgent',
    'StoryDiffusionAgent',
    'FreesoundSfxAgent',
    'FreesoundMusicAgent',
    'SlideshowVideoComposeAgent'
]