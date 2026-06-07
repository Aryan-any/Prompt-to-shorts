# 🎬 MM-StoryAgent — Prompt to YouTube Shorts, Fully Automated

> Give it a story idea. It writes the script, generates images, narrates it, scores it with music, cuts a vertical video, and uploads it to YouTube — all without you touching a single frame.

<div align="center">
  <img src="./assets/framework.png" alt="MM-StoryAgent Framework" style="width: 85%;" />
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)
![OpenAI](https://img.shields.io/badge/LLM-GPT--4.1-purple?style=flat-square)
![ElevenLabs](https://img.shields.io/badge/TTS-ElevenLabs-red?style=flat-square)
![Modal](https://img.shields.io/badge/Compute-Modal-black?style=flat-square)

</div>

---

## 📖 Overview

MM-StoryAgent is a multi-modal, multi-agent AI pipeline that transforms a raw story idea into a fully produced, narrated, and captioned YouTube Shorts video — and then uploads it directly to a YouTube channel. The system is built on top of the original research paper *"MM-StoryAgent: Immersive Narrated Storybook Video Generation with a Multi-Agent Paradigm across Text, Image and Audio"*, and this implementation extends it significantly with a production-grade automation layer: Google Sheets as a job queue, a Gradio control panel for batch operations, ElevenLabs for high-quality multilingual narration, Modal Labs for GPU-backed image and music inference, and a direct YouTube Data API upload pipeline.

The architecture is broken into clearly separated specialist agents — a story writer, an image generator, a speech synthesizer, a sound designer, a music composer, and a video compositor — all coordinated by a central `MMStoryAgent` orchestrator. Each agent is independently swappable via a YAML config file and a lightweight tool registry pattern. This means you can plug in your own image model, swap the TTS provider, or change the music backend without touching the core pipeline logic. The result is a system that is both research-friendly (modular, inspectable) and production-ready (automated, scalable, multi-language).

---

## ✨ Features

- **End-to-end automation** — From a one-line story prompt to a live YouTube Short, zero manual steps
- **Multi-agent story writing** — A QA dialogue loop between an *Asker* and an *Expert* LLM refines the story outline before a *Writer* agent turns it into per-scene narration text
- **Scene-coherent image generation** — DALL-E 3 or Flux (via Modal) generates one consistent storybook-style image per scene, guided by LLM-extracted character and scene descriptions
- **Multilingual narration** — ElevenLabs `eleven_multilingual_v2` produces high-quality speech for English and Hindi with per-character timestamp alignment for precise subtitle sync
- **AI music scoring** — MusicGen (via Modal) reads the full story and composes an original background music track tuned to the emotional tone of the narrative
- **Karaoke-style rolling subtitles** — Word-level timestamps from ElevenLabs are used to produce animated, chunk-highlighted subtitles (current word boxed in red/green) baked directly into the video
- **Slideshow video compositor** — MoviePy orchestrates pan/zoom Ken Burns effects, crossfade transitions, layered audio mixing (speech + music + sound at configurable ratios), and final 9:16 vertical encoding
- **Google Sheets job queue** — Stories are submitted via a Google Form/Sheet; the pipeline polls for rows marked `Ready to make = Yes` and marks them `done = yes` after upload
- **Gradio control panel** — A clean web UI lets you preview the pending job queue, trigger batch processing, and watch real-time progress
- **Pluggable tool registry** — Every agent is registered with `@register_tool("name")` and swapped in YAML config — no code changes required to switch models
- **Low-memory mode** — Sequential modality processing with CUDA cache clearing between stages makes it runnable on consumer hardware

---

## 📦 Tech Stack & Architecture

### Core Technologies

| Layer | Technology |
|---|---|
| **Orchestration** | Python, custom `MMStoryAgent` class |
| **LLM (Story + Prompts)** | OpenAI GPT-4.1 (`gpt-4.1-2025-04-14`) |
| **LLM (Music Prompts)** | Google Gemini (via `google-generativeai`) |
| **Image Generation** | DALL-E 3 (OpenAI API) / Flux (Modal Labs) |
| **Speech Synthesis** | ElevenLabs `eleven_multilingual_v2` / Google TTS |
| **Music Generation** | `facebook/musicgen-small` (Hugging Face / Modal) |
| **Video Composition** | MoviePy, OpenCV, NumPy |
| **Web UI** | Gradio 5.x |
| **Job Queue** | Google Sheets + gspread |
| **YouTube Upload** | Google YouTube Data API v3 |
| **Remote Compute** | Modal Labs (serverless GPU) |
| **Config Format** | YAML |

### Project Structure

```
prompt_to_shorts/
│
├── app.py                        # Gradio web UI + batch processing pipeline
├── run.py                        # CLI entry point (config-driven)
├── setup.py                      # Package installation
├── requirements.txt              # Python dependencies
├── .env.template                 # Environment variable reference
│
├── configs/
│   └── mm_story_agent.yaml       # Master pipeline configuration file
│
├── assets/
│   └── framework.png             # Architecture diagram
│
├── mm_story_agent/               # Core package
│   ├── __init__.py               # Exports MMStoryAgent
│   ├── base.py                   # Tool registry (@register_tool) + init_tool_instance
│   ├── mm_story_agent.py         # Main orchestrator: write → generate → compose
│   ├── prompts_en.py             # All LLM system prompts (story, image, music, review)
│   │
│   ├── modality_agents/          # One file per modality
│   │   ├── story_agent.py        # QA-dialogue multi-turn story writer
│   │   ├── image_agent.py        # DALL-E 3 / Flux image generation with role consistency
│   │   ├── speech_agent.py       # ElevenLabs + gTTS with timestamp CSV output
│   │   ├── music_agent.py        # MusicGen prompt generation + Modal inference
│   │   ├── sound_agent.py        # AudioLDM2 sound effects agent
│   │   ├── freesound_agent.py    # Freesound API for SFX and ambient music
│   │   ├── llm.py                # OpenAI and Gemini LLM wrappers
│   │   └── video_compose_agent.py # Full slideshow composer: zoom, pan, subtitles, audio mix
│   │
│   └── utils/
│       ├── llm_utils.py          # LLM config builder helper
│       ├── llm_logger.py         # Call logging utility
│       ├── llm_output_check.py   # Output validation helpers
│       └── import_utils.py       # Dynamic import helpers
│
├── story_eval/                   # Evaluation topics and rubrics
├── generated_stories/            # Output directory (auto-created)
│   └── example/
│       ├── image/                # Per-scene PNG images
│       ├── speech/               # Per-scene WAV audio + timestamp CSVs
│       ├── music/                # music.wav
│       ├── sound/                # Sound effect files
│       └── script_data.json      # Full story text (pages)
│
└── logs/                         # Runtime logs
```

### How the Pipeline Works

```
[Google Sheet / CLI prompt]
         │
         ▼
  ┌─────────────────┐
  │  Story Writer   │  ← GPT-4.1 (QA loop: Asker → Expert → Writer → Reviewer)
  │  (per scene)    │
  └────────┬────────┘
           │  pages[] (list of scene narration strings)
           ▼
  ┌─────────────────────────────────────────────────┐
  │           Parallel Modality Generation           │
  │  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐ │
  │  │  Image   │ │  Speech  │ │ Music  │ │Sound │ │
  │  │ (DALL-E3/│ │(ElevenLab│ │(MusicG │ │(Freq │ │
  │  │   Flux)  │ │ s/gTTS)  │ │ en)    │ │sound)│ │
  │  └──────────┘ └──────────┘ └────────┘ └──────┘ │
  └────────────────────┬────────────────────────────┘
                       │  images/, speech/ (WAV+CSV), music.wav
                       ▼
  ┌─────────────────────────────┐
  │     Video Compositor        │  ← MoviePy + OpenCV
  │  Ken Burns + crossfade      │
  │  Audio layering (3 tracks)  │
  │  Rolling karaoke subtitles  │
  │  9:16 vertical MP4 output   │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌──────────────────────────┐
  │   YouTube Upload (API)   │  ← Title, description, tags auto-generated by GPT-4.1
  │   Sheet row → done=yes   │
  └──────────────────────────┘
```

---

## ⚙️ Requirements & Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | 0 GB (CPU mode) | 8 GB (CUDA) |
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 |
| ImageMagick | 7.x | 7.1.1+ |
| FFmpeg | 4.x | 6.x |

> **Note:** Image and music generation are offloaded to Modal Labs serverless GPUs by default, so a local GPU is not required. If you switch to local SDXL or MusicGen, you'll want at least 8 GB VRAM.

### Required API Keys & Credentials

| Variable | Purpose | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Story writing, image prompts, metadata generation | [platform.openai.com](https://platform.openai.com) |
| `GOOGLE_API_KEY` | Gemini LLM for music prompt generation | [aistudio.google.com](https://aistudio.google.com) |
| `ELEVENLABS_API_KEY` | Neural TTS narration with timestamps | [elevenlabs.io](https://elevenlabs.io) |
| `STORY_DIR` | Output directory for generated assets | Set in `.env` |

### Additional Credential Files

| File | Purpose |
|---|---|
| `allinai_service_account.json` | Google Service Account for Sheets read/write access |
| `token_youtube1.json` | OAuth2 token for YouTube Data API uploads |
| `cosmic-quarter-*.json` | (Optional) Alternate GCP service account |

### Google Sheets Setup

Your Google Sheet must have these exact column headers in row 1:

```
Timestamp | Base Story | Genre | Ready to make | language | Youtube Link | done?
```

The pipeline reads rows where `Ready to make = Yes` and `done? ≠ yes`, processes them, and writes the YouTube URL + marks `done? = yes` when complete.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/prompt_to_shorts.git
cd prompt_to_shorts
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Install System Dependencies

**ImageMagick** (required for subtitle text rendering):

```bash
# Windows — download and install from:
# https://imagemagick.org/script/download.php#windows
# Install to: C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\

# Ubuntu / Debian
sudo apt-get install imagemagick

# macOS
brew install imagemagick
```

**FFmpeg** (required by MoviePy):

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 5. Configure Environment Variables

```bash
cp .env.template .env
```

Open `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ELEVENLABS_API_KEY=...
STORY_DIR=generated_stories/example
```

### 6. Set Up Google Credentials

Place your Google Service Account JSON file in the project root:

```bash
# Rename your downloaded service account key to:
allinai_service_account.json
```

For YouTube uploads, authenticate once using OAuth2:

```bash
# First-time setup — this opens a browser for Google OAuth consent
python -c "
from google_auth_oauthlib.flow import InstalledAppFlow
flow = InstalledAppFlow.from_client_secrets_file('credentials_oauth_youtube.json', ['https://www.googleapis.com/auth/youtube.upload'])
creds = flow.run_local_server(port=0)
with open('token_youtube1.json', 'w') as f:
    f.write(creds.to_json())
print('Token saved.')
"
```

### 7. Set Up Modal Labs (for GPU inference)

```bash
pip install modal
modal token new
```

Deploy the image and music generation apps to Modal:

```bash
# Deploy image generation (Flux)
modal deploy modal_apps/image.py  # adjust path to your Modal app files

# Deploy music generation (MusicGen)
modal deploy modal_apps/musicgen.py
```

---

## 🛠️ Usage Examples

### Option A: Run from the Gradio Web UI (Recommended)

Launch the control panel:

```bash
python app.py
```

This starts a Gradio server at `http://localhost:7000`. The UI gives you:
- **Pending count** — How many stories in your Google Sheet are queued
- **Preview Stories** — See the first 10 pending stories before processing
- **Start Video Production** — Triggers batch processing with a live progress bar

### Option B: Run from the Command Line (Config-driven)

Edit `configs/mm_story_agent.yaml` to set your story, then run:

```bash
python run.py -c configs/mm_story_agent.yaml
```

Override specific settings at runtime:

```bash
# Use Gemini instead of OpenAI for story writing
python run.py -c configs/mm_story_agent.yaml --llm gemini

# Use Google TTS instead of ElevenLabs
python run.py -c configs/mm_story_agent.yaml --tts gtts

# Run in low-memory mode (default: enabled)
python run.py -c configs/mm_story_agent.yaml --low-memory
```

### Option C: Use the Python API Directly

```python
from mm_story_agent import MMStoryAgent

config = {
    "sample_rate": 16000,
    "story_dir": "generated_stories/my_story",
    
    "story_writer": {
        "tool": "qa_outline_story_writer",
        "cfg": {
            "max_conv_turns": 3,
            "num_outline": 4,
            "temperature": 0.5,
            "llm": "openai",
            "model_name": "gpt-4.1-2025-04-14",
        },
        "params": {
            "story_topic": "A fox who learns honesty is the best strategy",
            "main_role": "The clever fox",
            "scene": "A moonlit forest clearing where animals hold a council",
            "base_story": "Once there was a fox who survived by lying...",
            "story_lang": "english",
            "video_style": "youtube shorts",
        },
    },
    
    "speech_generation": {
        "tool": "elevenlabs",
        "cfg": {"sample_rate": 16000},
        "params": {"lang": "english"},
    },
    
    "image_generation": {
        "tool": "story_diffusion_t2i",
        "cfg": {
            "device": "cpu",
            "num_turns": 3,
            "llm": "openai",
            "model": "dall-e-3",
            "height": 1024,
            "width": 1024,
        },
        "params": {"style_name": "Storybook", "quality": "standard"},
    },
    
    "music_generation": {
        "tool": "musicgen_t2m",
        "cfg": {
            "device": "cpu",
            "llm_type": "gemini",
            "num_turns": 3,
            "model_name": "facebook/musicgen-small",
        },
        "params": {"duration": 30},
    },
    
    "video_composition": {
        "tool": "slideshow_video_compose",
        "cfg": {
            "fps": 10,
            "audio_sample_rate": 16000,
            "audio_codec": "mp3",
            "fade_duration": 0.1,
            "slide_duration": 0.1,
            "zoom_speed": 0.5,
            "move_ratio": 0.95,
            "sound_volume": 0.2,
            "music_volume": 0.15,
            "bg_speech_ratio": 0.4,
            "caption_config": {
                "area_height": 100,
                "max_length": 100,
                "fontsize": 24,
                "color": "white",
                "font": "Arial",
            },
        },
        "params": {},
    },
    
    "system": {
        "low_memory_mode": True,
        "batch_size": 1,
        "max_parallel_processes": 1,
    },
}

agent = MMStoryAgent(low_memory_mode=True)
agent.call(config, video_title="The Honest Fox")
```

### Registering a Custom Agent

You can drop in your own implementation for any modality. Here's how to replace the TTS with a custom provider:

```python
# my_custom_tts.py
from typing import Dict
from pathlib import Path
from mm_story_agent.base import register_tool

@register_tool("my_tts")
class MyTTSAgent:
    
    def __init__(self, cfg: Dict):
        self.api_key = cfg.get("api_key")
        self.sample_rate = cfg.get("sample_rate", 16000)
    
    def call(self, params: Dict):
        pages = params["pages"]          # list of story text strings
        save_path = Path(params["save_path"])
        lang = params.get("lang", "english")
        
        for idx, page_text in enumerate(pages):
            output_file = save_path / f"p{idx + 1}.wav"
            # Your TTS logic here
            self._synthesize(page_text, lang, output_file)
        
        return {"modality": "speech"}
    
    def _synthesize(self, text, lang, output_path):
        # Your implementation
        pass
```

Then in your YAML config:

```yaml
speech_generation:
    tool: my_tts
    cfg:
        api_key: "your-api-key"
        sample_rate: 16000
    params:
        lang: english
```

That's it. No other code changes needed.

---

## 🧩 Configuration Reference

The YAML config controls every agent in the pipeline. Here's a full annotated reference:

```yaml
# Global settings
sample_rate: &sample_rate 16000     # Audio sample rate (Hz)
image_height: &image_height 512     # Output image height
image_width: &image_width 512       # Output image width
story_dir: &story_dir generated_stories/example  # Where all assets are saved

# Story Writer
story_writer:
    tool: qa_outline_story_writer   # Registered tool name
    cfg:
        max_conv_turns: 3           # QA dialogue rounds before writing
        num_outline: 4              # Number of chapters/scenes to outline
        temperature: 0.5            # LLM temperature
        llm: openai                 # "openai" or "gemini"
        model_name: gpt-4.1-2025-04-14
    params:
        story_topic: "..."          # The core moral or subject of the story
        main_role: "..."            # Primary character description
        scene: "..."                # Key scene to build around
        base_story: |               # Optional seed story (improves quality significantly)
            Your story text here...
        story_lang: english         # "english" or "hindi"
        video_style: youtube shorts

# Speech (TTS)
speech_generation:
    tool: elevenlabs               # "elevenlabs" or "gtts"
    cfg:
        sample_rate: *sample_rate
    params:
        lang: english              # "english" or "hindi"

# Image Generation
image_generation:
    tool: story_diffusion_t2i
    cfg:
        device: cpu                # "cpu" or "cuda"
        num_turns: 3               # Review/refinement rounds for image prompts
        llm: openai
        model: dall-e-3
        height: 1024
        width: 1024
    params:
        style_name: Storybook      # Art style (see supported styles below)
        quality: standard          # "standard" or "hd"

# Music Generation
music_generation:
    tool: musicgen_t2m
    cfg:
        device: cpu
        llm_type: gemini           # LLM used to write the music prompt
        num_turns: 3
        model_name: facebook/musicgen-small
    params:
        duration: 30               # Music duration in seconds

# Video Compositor
video_composition:
    tool: slideshow_video_compose
    cfg:
        fps: 10
        fade_duration: 0.1         # Crossfade between scenes (seconds)
        slide_duration: 0.1        # Slide transition duration
        zoom_speed: 0.5            # Ken Burns zoom intensity
        move_ratio: 0.95           # Pan distance as fraction of frame width
        sound_volume: 0.2          # SFX track volume (0.0–1.0)
        music_volume: 0.15         # Music track volume
        bg_speech_ratio: 0.4       # Speech background fade ratio
        caption_config:
            fontsize: 24
            color: white
            font: Arial
            area_height: 100
            max_length: 100        # Characters per subtitle chunk
    params: {}

# System
system:
    low_memory_mode: true          # Sequential processing, clears CUDA cache between stages
    batch_size: 1
    max_parallel_processes: 1
```

**Supported image style names:**
`Storybook`, `Japanese Anime`, `Digital/Oil Painting`, `Pixar/Disney Character`, `Photographic`, `Comic book`, `Line art`, `Black and White Film Noir`, `Isometric Rooms`

---

## 📊 Story Quality Evaluation

The pipeline's story writing quality was benchmarked against direct LLM prompting across four topic categories using GPT-4 as evaluator (rubric: Attractiveness, Warmth, Education).

| Topic | Method | Attractiveness | Warmth | Education | **Average** |
|---|---|---|---|---|---|
| Self-growing | Direct | 3.68 | 4.42 | 4.84 | 4.31 |
| | **Story Agent** | **4.10** | **4.50** | **4.80** | **4.47** |
| Family & Friendship | Direct | 3.94 | 5.00 | 4.72 | 4.55 |
| | **Story Agent** | **4.36** | **4.80** | **4.92** | **4.69** |
| Environments | Direct | 4.00 | 4.62 | 4.92 | 4.51 |
| | **Story Agent** | **4.44** | **4.68** | **4.86** | **4.66** |
| Knowledge Learning | Direct | 4.46 | 4.14 | 4.86 | 4.49 |
| | **Story Agent** | **4.84** | **4.52** | **4.90** | **4.75** |
| **All Topics** | Direct | 4.02 | 4.55 | 4.84 | 4.47 |
| | **Story Agent** | **4.44** | **4.63** | **4.87** | **4.65** |

The multi-turn QA dialogue loop consistently outperforms single-shot prompting, especially on Attractiveness — the dimension most affected by narrative creativity and structure.

---

## 🧪 Running Tests

```bash
# Verify your environment is correctly set up
python check_video.py

# Run a minimal end-to-end test (uses a short story, skips YouTube upload)
python run.py -c configs/mm_story_agent.yaml --low-memory

# Check that Modal deployments are reachable
python -c "
import modal
music_gen = modal.Cls.lookup('musicgen', 'MusicGen')
print('MusicGen deployment found:', music_gen)
flux = modal.Cls.lookup('image', 'FluxImage')
print('FluxImage deployment found:', flux)
"

# Test Google Sheets connection
python -c "
from app import init_google_sheets, get_pending_videos
sheet, _ = init_google_sheets()
pending, full = get_pending_videos(sheet)
print(f'Connected. Pending videos: {len(pending)}')
"
```

---

## 🗂️ Output Files

After a successful run, the `generated_stories/example/` directory contains:

```
generated_stories/example/
├── image/
│   ├── p1.png          # Scene 1 image (1080×1920 for Shorts)
│   ├── p2.png
│   └── ...
├── speech/
│   ├── p1.wav          # Scene 1 narration audio
│   ├── p1.csv          # Per-character timestamp alignment
│   └── ...
├── music/
│   └── music.wav       # AI-composed background track
├── sound/
│   └── (optional SFX files)
├── script_data.json    # Full structured story (all pages)
└── <video_title>.mp4   # Final composed vertical video
```

For English stories, a second output with rolling karaoke subtitles is also produced:

```
<timestamp>chunked_rolling_subs.mp4
```

---

## 🌐 Demo

Watch a sample output video generated entirely by this pipeline:

<div align="center">
  <a href="https://www.youtube.com/watch?v=2HXGrA8mg90" target="_blank">
    <img src="https://res.cloudinary.com/marcomontalbano/image/upload/v1723627863/video_to_markdown/images/youtube--2HXGrA8mg90-c05b58ac6eb4c4700831b2b3070cd403.jpg" alt="MM-StoryAgent demo" style="width: 60%;"/>
  </a>
</div>

---

## 🔧 Troubleshooting

**ImageMagick not found (Windows)**

The video compositor auto-detects ImageMagick in common install paths. If it fails, set it manually at the top of `video_compose_agent.py`:

```python
mpconfig.change_settings({"IMAGEMAGICK_BINARY": r"C:\Your\Path\To\magick.exe"})
```

**JSON parse failures in story/metadata generation**

The pipeline has a built-in retry loop (up to 5 attempts) with a `json_reformatter` agent that calls GPT-4 to fix malformed JSON. If it still fails, try increasing `temperature` slightly (0.6–0.7) in the story writer config.

**Modal `lookup` errors**

Make sure your Modal apps are deployed and your Modal token is active:

```bash
modal token new
modal deploy your_modal_app.py
```

**ElevenLabs voice ID mismatch**

The voice IDs in `speech_agent.py` are hardcoded per language. Update them to match voices on your ElevenLabs account:

```python
if lang == "english":
    voice_id = "your_english_voice_id"
elif lang == "hindi":
    voice_id = "your_hindi_voice_id"
```

---

## 🤝 Contributing

Contributions are welcome. Here's the quickest way to get involved:

1. **Fork** the repository and create a feature branch: `git checkout -b feature/my-new-agent`
2. **Add your agent** using the `@register_tool` decorator pattern (see [Custom Agent](#-usage-examples) section above)
3. **Test** your agent end-to-end with a short story config
4. **Open a Pull Request** with a clear description of what your agent does and which modality it targets

If you find a bug or want to request a feature, please [open an issue](../../issues) with as much detail as possible (error logs, config used, OS, Python version).

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software for both personal and commercial purposes, as long as you include the original copyright notice and license text. See the [LICENSE](./LICENSE) file for full terms.

---

<div align="center">
  <sub>Built with GPT-4.1 · ElevenLabs · Modal Labs · MoviePy · Gradio</sub>
</div>
