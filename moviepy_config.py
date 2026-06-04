import os
from moviepy.config import change_settings

def find_imagemagick_binary():
    """Find ImageMagick binary path"""
    # Common ImageMagick installation paths on Windows
    possible_paths = [
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe",
        r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16\magick.exe",
    ]
    
    # Check environment variable first
    if os.getenv('IMAGEMAGICK_BINARY'):
        return os.getenv('IMAGEMAGICK_BINARY')
    
    # Check common paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Default to 'magick' and hope it's in PATH
    return 'magick'

# Set ImageMagick binary path
IMAGEMAGICK_BINARY = find_imagemagick_binary()
print(f"Using ImageMagick binary: {IMAGEMAGICK_BINARY}")

# Configure MoviePy
change_settings({
    "IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY
}) 