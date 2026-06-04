import os
from pathlib import Path

# Check both possible locations
output_paths = [
    Path("generated_stories/final_output/final_video.mp4"),
    Path("generated_stories/example/final_video.mp4")
]

for path in output_paths:
    if path.exists():
        print(f"FOUND VIDEO: {path.absolute()}")
        print(f"Video size: {path.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"Created at: {os.path.getctime(path)}")
        print(f"Modified at: {os.path.getmtime(path)}")
    else:
        print(f"Video not found at: {path.absolute()}")

# Check if the directories exist and list their contents
dirs_to_check = [
    Path("generated_stories/final_output"),
    Path("generated_stories/example")
]

for directory in dirs_to_check:
    if directory.exists():
        print(f"\nContents of {directory}:")
        for item in directory.iterdir():
            if item.is_file():
                print(f"  {item.name} - {item.stat().st_size / 1024:.2f} KB")
            else:
                print(f"  {item.name}/ (directory)")
    else:
        print(f"\nDirectory not found: {directory}") 