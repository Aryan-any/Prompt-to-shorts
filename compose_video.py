from pathlib import Path
from mm_story_agent import MMStoryAgent
import yaml
import json
import os
import shutil
import sys
import traceback
video_title = "test_case12"
try:
    # Load the config
    with open('configs/mm_story_agent.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Initialize the agent
    agent = MMStoryAgent()

    # Source directory where assets are stored
    source_dir = Path("generated_stories/example")
    print(f"Source directory: {source_dir.absolute()}")

    # Try to load the original story text
    story_text = []
    try:
        # Try to load from script_data.json first
        if (source_dir / "script_data.json").exists():
            with open(source_dir / "script_data.json", "r") as f:
                script_data = json.load(f)
                story_text = [page["story"] for page in script_data["pages"]]
                print("Loaded story text from script_data.json")
        else:
            # Try to load from text files
            text_dir = source_dir / "text"
            if text_dir.exists():
                text_files = sorted(text_dir.glob("p*.txt"))
                for txt_file in text_files:
                    with open(txt_file, "r", encoding='utf-8') as f:
                        story_text.append(f.read().strip())
                print("Loaded story text from text files")
    except Exception as e:
        print(f"Error loading story text: {e}")
        traceback.print_exc()

    if not story_text:
        print("Warning: Could not load original story text!")

    # # Create a new directory for the final video
    # output_dir = Path("generated_stories/final_output")
    # output_dir.mkdir(exist_ok=True, parents=True)
    # print(f"Output directory created at: {output_dir.absolute()}")

    # # Copy required assets to new directory
    # print("\nCopying assets to new directory...")
    # for subdir in ['image', 'sound', 'speech', 'music']:
    #     src_dir = source_dir / subdir
    #     dst_dir = output_dir / subdir
    #     if src_dir.exists():
    #         print(f"Copying {subdir} files...")
    #         shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    # Update config to use new directory
    config["story_dir"] = str(source_dir)

    # Try to get pages from image files
    image_dir = source_dir / "image"
    if image_dir.exists():
        image_files = sorted(list(image_dir.glob('p*.png')))
        if image_files:
            print(f"\nFound {len(image_files)} image files")
            
            # Use original story text if available, otherwise use generic pages
            if story_text and len(story_text) == len(image_files):
                pages = story_text
                print("Using original story text for captions")
            else:
                pages = [f"Page {i+1}" for i in range(len(image_files))]
                print("Using generic page numbers for captions")
            
            # Create script data
            script_data = {"pages": [{"story": page} for page in pages]}
            
            print("\nChecking required files:")
            for idx in range(len(pages)):
                print(f"\nPage {idx+1}:")
                print(f"Image exists: {(image_dir / f'p{idx+1}.png').exists()}")
                print(f"Speech exists: {(source_dir / 'speech' / f'p{idx+1}.wav').exists()}")
                print(f"Sound exists: {(source_dir / 'sound' / f'p{idx+1}.wav').exists()}")
            print(f"Music exists: {(source_dir / 'music' / 'music.wav').exists()}")
            
            try:
                print("\nStarting video composition...")
                result = agent.compose_video(config, pages, script_data, video_title= video_title)
                
                if result is not None:
                    output_video = source_dir / "final_video.mp4"
                    if output_video.exists():
                        print(f"\nSuccess! Video saved at: {output_video.absolute()}")
                        # Try to copy back to original directory
                        try:
                            shutil.copy2(output_video, source_dir / "final_video.mp4")
                            print(f"Video also copied to: {source_dir / 'final_video.mp4'}")
                        except Exception as e:
                            print(f"Could not copy video back to source directory: {e}")
                    else:
                        print("\nError: Video file not found after composition!")
                else:
                    print("\nError: Video composition returned None!")
                    
            except Exception as e:
                print(f"\nError during video composition: {str(e)}")
                traceback.print_exc()
        else:
            print("No image files found with pattern p*.png")
    else:
        print(f"Image directory not found at {image_dir}")

except Exception as e:
    print(f"Fatal error: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 