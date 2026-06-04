import time
import json
from pathlib import Path
import re
import torch.multiprocessing as mp
import torch
mp.set_start_method("spawn", force=True)

from .base import init_tool_instance


class MMStoryAgent:

    def __init__(self, low_memory_mode=True) -> None:
        self.modalities = ["image", "sound", "speech", "music"]
        self.low_memory_mode = low_memory_mode

    def cleanup(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def call_modality_agent(self, modality, agent, params, return_dict):
        try:
            print(f"Generating {modality} assets...")
            result = agent.call(params)
            if result:  # Check if result is not None
                print(f"Successfully generated {modality} assets")
                return_dict[modality] = result
            else:
                print(f"No results generated for {modality}")
                return_dict[modality] = {"error": "No results generated"}
        except Exception as e:
            print(f"Error generating {modality} assets: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.__dict__}")
            return_dict[modality] = {"error": str(e)}

    def write_story(self, config):
        print("Starting story generation...")
        cfg = config["story_writer"]
        print(f"Using config: {cfg}")
        story_writer = init_tool_instance(cfg)
        print("Story writer initialized")
        pages = story_writer.call(cfg["params"])
        return pages
    
    def generate_modality_assets(self, config, pages):
        script_data = {"pages": [{"story": page} for page in pages]}
        story_dir = Path(config["story_dir"])
        images = None
        print(f"\n\nScript data:\n {script_data}\n")

        # Create subdirectories for each modality
        for sub_dir in self.modalities:
            (story_dir / sub_dir).mkdir(exist_ok=True, parents=True)

        return_dict = {}
        
        # Run modalities sequentially instead of in parallel
        for modality in self.modalities:
            try:
                print(f"\nProcessing {modality}...")
                agent = init_tool_instance(config[modality + "_generation"])
                params = config[modality + "_generation"]["params"].copy()
                params.update({
                    "pages": pages,
                    "save_path": story_dir / modality
                })
                
                # Clear CUDA cache before each modality
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                result = agent.call(params)
                return_dict[modality] = result
                print(f"{modality} generation completed")
                
            except Exception as e:
                print(f"Error in {modality} generation: {str(e)}")
                return_dict[modality] = {"error": str(e)}

        try:
            print(f"Writing script_data.json to: {story_dir / 'script_data.json'}")
            print(f"Script data content (truncated): {str(script_data)[:500]}")
            with open(story_dir / "script_data.json", "w", encoding="utf-8") as writer:
                json.dump(script_data, writer, ensure_ascii=False, indent=4)
            print("Successfully wrote script_data.json")
        except Exception as e:
            print(f"Error writing script_data.json: {e}")
            import traceback
            traceback.print_exc()

        return images
    
    def compose_video(self, config, pages, script_data,video_title):
        try:
            print("Starting video composition...")
            video_cfg = config.get("video_composition", {})
            print(f"Video config: {video_cfg}")  # Debug print
            
            if not video_cfg:
                print("Warning: No video composition configuration found!")
                return None
            
            video_composer = init_tool_instance(video_cfg)
            print("Video composer initialized")  # Debug print
            
            # Prepare parameters for video composition
            video_title = re.sub(r'[<>:"/\\|?*]', '', video_title)
            story_dir = Path(config["story_dir"])
            vid_path = str(video_title) + ".mp4"
            save_path = story_dir / vid_path
            music_path = story_dir / "music" / "music.wav"
            
            print(f"Story directory: {story_dir}")  # Debug print
            print(f"Save path: {save_path}")  # Debug print
            print(f"Music path exists: {music_path.exists()}")  # Debug print
            
            # Get captions from script data
            captions = [page["story"] for page in script_data["pages"]]
            print(f"Number of captions: {len(captions)}")  # Debug print
            
            params = {
                "story_dir": story_dir,
                "save_path": save_path,
                "captions": captions,
                "music_path": music_path if music_path.exists() else None,
                "num_pages": len(pages)
            }
            
            # Add any additional parameters from config
            params.update(video_cfg.get("params", {}))
            print(f"Final video params: {params}")  # Debug print
            
            result = video_composer.call(params)
            print("Video composition completed successfully")
            return result
            
        except Exception as e:
            print(f"Error in video composition: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            return None
        
    def clear_directory(self,directory_path):
        """Check if directory exists and clear it if not empty."""
        directory = Path(directory_path)
        if directory.exists():
            files = list(directory.glob('*'))
            if files:
                print(f"Clearing {len(files)} files from {directory}")
                for file in files:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil
                        shutil.rmtree(file)
                print(f"Directory {directory} has been cleared")
            else:
                print(f"Directory {directory} is already empty")
        else:
            print(f"Directory {directory} does not exist")
            directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory {directory}")

    def call(self, config, video_title):
        try:
            # Create the main story directory first
            story_dir = Path(config["story_dir"])
            story_dir.mkdir(exist_ok=True, parents=True)
            story_dir1 = Path(story_dir)
            sound_dir = story_dir1 / "sound"
            image_dir = story_dir1 / "image"
            speech_dir = story_dir1 / "speech"

            

            self.clear_directory(sound_dir)
            self.clear_directory(image_dir)
            self.clear_directory(speech_dir)

            # Process one modality at a time
            pages = self.write_story(config)
            self.cleanup()
            
            images = self.generate_modality_assets(config, pages)
            self.cleanup()
            
            script_data = {"pages": [{"story": page} for page in pages]}
            self.compose_video(config, pages, script_data, video_title)
            self.cleanup()
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")



