from pathlib import Path
from typing import List, Dict
import json
import os

import torch
import soundfile as sf
from diffusers import AudioLDM2Pipeline

from mm_story_agent.prompts_en import story_to_sound_reviser_system, story_to_sound_review_system
from mm_story_agent.base import register_tool, init_tool_instance


class AudioLDM2Synthesizer:

    def __init__(self, device: str = 'cpu') -> None:
        self.device = device
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.pipe = AudioLDM2Pipeline.from_pretrained(
                "cvssp/audioldm2",
                torch_dtype=torch.float32,
                use_safetensors=True,
                local_files_only=False,
                token=os.getenv('HF_TOKEN'),
                low_cpu_mem_usage=True
            )
            self.pipe = self.pipe.to(self.device)
        except Exception as e:
            print(f"Error initializing AudioLDM2: {str(e)}")
            raise
    
    def call(
        self,
        prompts: List[str],
        n_candidate_per_text: int = 3,
        seed: int = 0,
        guidance_scale: float = 3.5,
        ddim_steps: int = 100,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        audios = self.pipe(
            prompts, 
            num_inference_steps=ddim_steps, 
            audio_length_in_s=10.0,
            guidance_scale=guidance_scale,
            generator=generator,
            num_waveforms_per_prompt=n_candidate_per_text).audios
        
        audios = audios[::n_candidate_per_text]

        return audios


@register_tool("audioldm2_t2a")
class AudioLDM2Agent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def generate_sound_prompt_from_story(self, pages):
        sound_prompt_reviser = init_tool_instance({
            "tool": self.cfg.get("llm", "gemini"),
            "cfg": {
                "system_prompt": story_to_sound_reviser_system,
                "track_history": False
            }
        })
        
        sound_prompt_reviewer = init_tool_instance({
            "tool": self.cfg.get("llm", "gemini"),
            "cfg": {
                "system_prompt": story_to_sound_review_system,
                "track_history": False
            }
        })

        
        review = ""
        prompt = ""
                
        sound_prompts = []
        for page in pages:
            for turn in range(self.cfg.get("max_turns", 3)):
                prompt, _ = sound_prompt_reviser.call(json.dumps({
                    "story": page,
                    "previous_result": prompt,
                    "improvement_suggestions": review,
                }, ensure_ascii=False))
                print(f"\n\nSound prompt reviser result {turn + 1}/{self.cfg.get('max_turns', 3)}:\n {prompt}\n")
                review, success = sound_prompt_reviewer.call(json.dumps({
                    "story_content": page,
                    "sound_description": prompt
                }, ensure_ascii=False))
                print(f"\n\nSound prompt reviewer result {turn + 1}/{self.cfg.get('max_turns', 3)}:\n {review}\n")
                if review == "Check passed.":
                    break
            sound_prompts.append(prompt)
        return sound_prompts

    def call(self, params: Dict):
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        save_dir = Path(save_path)
        if save_dir.exists():
            for file in save_dir.glob("*.wav"):
                file.unlink()
        else:
            save_dir.mkdir(parents=True, exist_ok=True)
        sound_prompts = self.generate_sound_prompt_from_story(pages)
        print(f"\n\nSound prompts:\n {sound_prompts}\n")
        save_paths = []
        forward_prompts = []
        save_path = Path(save_path)
        
        for idx, prompt in enumerate(sound_prompts):
            if prompt != "No sounds.":
                save_paths.append(save_path / f"p{idx + 1}.wav")
                forward_prompts.append(prompt)
        
        generation_agent = AudioLDM2Synthesizer(device=self.cfg.get("device", "cpu"))
        if len(forward_prompts) > 0:
            print(f"\n\nSynthesizing sounds with prompts:\n {forward_prompts}\n")
            sounds = generation_agent.call(
                forward_prompts,
                n_candidate_per_text=params.get("n_candidate_per_text", 3),
                seed=params.get("seed", 0),
                guidance_scale=params.get("guidance_scale", 3.5),
                ddim_steps=params.get("ddim_steps", 100),
            )
            for sound, path in zip(sounds, save_paths):
                sf.write(path.__str__(), sound, self.cfg["sample_rate"])
                print(f"\n\nSaved sound to {path}")
        return {
            "prompts": sound_prompts,
        }

