from pathlib import Path
from typing import List, Union
import random
import re
from datetime import timedelta
import traceback
import os
import sys
import pandas as pd 
import numpy as np
from tqdm import trange
import librosa
import cv2
import math

# Import moviepy config first
import moviepy.config as mpconfig

def ensure_imagemagick():
    """Ensure ImageMagick is properly configured"""
    if os.name == 'nt':  # Windows
        # Try to find ImageMagick installation
        possible_paths = [
            r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe",
            r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
            r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16\magick.exe",
        ]
        
        imagemagick_path = None
        for path in possible_paths:
            if os.path.exists(path):
                imagemagick_path = path
                break
        
        if imagemagick_path:
            mpconfig.change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
            print(f"Using ImageMagick from: {imagemagick_path}")
        else:
            print("Warning: ImageMagick not found in common locations")
            mpconfig.change_settings({"IMAGEMAGICK_BINARY": "magick"})

# Call this function before any MoviePy operations
ensure_imagemagick()

# MoviePy imports
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx.all import audio_loop
from moviepy.video.fx.all import resize
from moviepy.video.tools.subtitles import SubtitlesClip

from ..base import register_tool

# Update the class references to use mpy instead
ImageClip = mpy.ImageClip
AudioFileClip = mpy.AudioFileClip
CompositeAudioClip = mpy.CompositeAudioClip
CompositeVideoClip = mpy.CompositeVideoClip
ColorClip = mpy.ColorClip
VideoFileClip = mpy.VideoFileClip
VideoClip = mpy.VideoClip
TextClip = mpy.TextClip
concatenate_audioclips = mpy.concatenate_audioclips
concatenate_videoclips = mpy.concatenate_videoclips


def generate_srt(timestamps: List,
                 captions: List,
                 save_path: Union[str, Path],
                 max_single_length: int = 30):
    """Generate SRT subtitle file"""
    def format_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        millis = int((td.total_seconds() - total_seconds) * 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
    
    # Ensure save_path is Path object
    save_path = Path(save_path)
    
    print(f"\nGenerating SRT file at: {save_path}")
    print(f"Number of captions: {len(captions)}")
    print(f"Number of timestamps: {len(timestamps)}")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, (caption, (start, end)) in enumerate(zip(captions, timestamps), 1):
            print(f"\nProcessing caption {i}:")
            print(f"Caption text: {caption}")
            print(f"Start time: {start}, End time: {end}")
            
            # Split caption if too long
            words = caption.split()
            lines = []
            current_line = []
            
            for word in words:
                if len(' '.join(current_line + [word])) <= max_single_length:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            print(f"Split into lines: {lines}")
            
            # Write SRT entry
            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write('\n'.join(lines) + '\n\n')
    
    print(f"SRT file generated successfully at {save_path}")


def add_caption(captions: List,
                srt_path: Union[str, Path],
                timestamps: List,
                video_clip: VideoClip,
                max_single_length: int = 30,
                **caption_config):
    """Add captions to video clip"""
    try:
        print("\nStarting add_caption function")
        print(f"Video clip size: {video_clip.w}x{video_clip.h}")
        print(f"Caption config: {caption_config}")
        
        # Generate SRT file
        generate_srt(timestamps, captions, srt_path, max_single_length)
        
        # Verify SRT file was created
        if not Path(srt_path).exists():
            print(f"Error: SRT file was not created at {srt_path}")
            return video_clip
            
        # Create text clip generator with proper configuration
        generator = lambda txt: TextClip(
            txt,
            font=caption_config.get('font', 'Arial'),
            fontsize=caption_config.get('fontsize', 18),
            color=caption_config.get('color', 'black'),
            stroke_color=caption_config.get('stroke_color', 'white'),
            stroke_width=caption_config.get('stroke_width', 0),
            method='caption',
            size=(video_clip.w, None)
        )
        
        print("Creating subtitles clip...")
        # Create subtitles clip
        subtitles = SubtitlesClip(str(srt_path), generator)
        
        # Position subtitles at the bottom of the video
        area_height = caption_config.get('area_height', 100)
        y_position = video_clip.h - area_height + 20
        print(f"Positioning subtitles at y={y_position} (video height={video_clip.h}, area_height={area_height})")
        
        captioned_clip = CompositeVideoClip([
            video_clip,
            subtitles.set_position(('center', y_position))
        ])
        
        print("Successfully created captioned clip")
        return captioned_clip
        
    except Exception as e:
        print(f"Error in add_caption: {str(e)}")
        traceback.print_exc()
        # Return original clip if caption addition fails
        return video_clip


def split_keep_separator(text: str, separators: str) -> List[str]:
    """Split text keeping separators"""
    pattern = f"([{separators}])"
    return [s for s in re.split(pattern, text) if s.strip()]


def split_caption(caption: str, max_length: int = 30) -> str:
    """Split caption into multiple lines if too long"""
    if not caption:
        return ""
        
    # First try to split by punctuation
    sentences = split_keep_separator(caption, r'[.!?。！？]')
    if not sentences:
        sentences = [caption]
        
    lines = []
    current_line = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            word_len = len(word)
            if current_length + word_len + 1 <= max_length:
                current_line.append(word)
                current_length += word_len + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len + 1
                
    if current_line:
        lines.append(' '.join(current_line))
        
    return '\n'.join(lines)

def rolling_subtitle_data(words_and_times, max_words=5):
    """
    Build a list of ((start, end), text) tuples suitable for MoviePy's SubtitlesClip,
    using a rolling window of exactly max_words words, resetting after each chunk.
    The newest word will be in UPPERCASE.
    
    :param words_and_times: List of (word, start_time, end_time) for each word.
    :param max_words: Maximum number of words in the rolling window.
    :return: A list of ((start, end), text) intervals for MoviePy.
    """
    subtitles = []
    for i, (word, start_time, end_time) in enumerate(words_and_times):
        # Determine the time when this subtitle should stop
        if i + 1 < len(words_and_times):
            _, next_start, _ = words_and_times[i + 1]
            display_end = next_start
        else:
            display_end = end_time

        # Calculate which chunk this word belongs to and its position within the chunk
        chunk_number = i // max_words
        position_in_chunk = i % max_words
        
        # Get the start index for this chunk
        chunk_start = chunk_number * max_words
        
        # Get words for this chunk (up to the current word)
        visible_words = [w for (w, _, _) in words_and_times[chunk_start:chunk_start + position_in_chunk + 1]]
        
        # "Highlight" the newest word in uppercase
        visible_words[-1] = visible_words[-1].upper()
        
        # Join them all into a single subtitle line
        subtitle_text = " ".join(visible_words)
        
        # Append to our list in MoviePy's expected format
        subtitles.append(((start_time, display_end), subtitle_text))

    return subtitles

def rolling_subtitle_data_chunked(words_and_times, max_words=5, max_chunk_length=25):
    """
    Build a list of ((start, end), text) tuples suitable for MoviePy's SubtitlesClip,
    using chunks of max_words, respecting max_chunk_length (total chars).
    The current word in the chunk is highlighted.
    """
    subtitles = []

    current_chunk_words = []
    current_chunk_start_time = None
    chunk_char_count = 0
    chunk_start_idx = 0

    for i, (word, start_time, end_time) in enumerate(words_and_times):
        # If starting new chunk
        if not current_chunk_words:
            current_chunk_start_time = start_time
            chunk_char_count = 0
            chunk_start_idx = i

        # Check if adding this word would overflow chunk
        word_len = len(word)
        if (len(current_chunk_words) >= max_words) or (chunk_char_count + word_len > max_chunk_length):
            # Close the current chunk for the last word
            # The "previous" word stays highlighted until next chunk
            for j in range(chunk_start_idx, i):
                chunk_words_display = [w for (w, _, _) in words_and_times[chunk_start_idx:j+1]]
                chunk_words_display[-1] = chunk_words_display[-1].upper()  # highlight current word

                subtitle_text = " ".join(chunk_words_display)
                subtitle_start = words_and_times[j][1]
                if j + 1 < len(words_and_times):
                    subtitle_end = words_and_times[j+1][1]
                else:
                    subtitle_end = words_and_times[j][2]

                subtitles.append(((subtitle_start, subtitle_end), subtitle_text))

            # Start new chunk
            current_chunk_words = []
            current_chunk_start_time = start_time
            chunk_char_count = 0
            chunk_start_idx = i

        # Add word to current chunk
        current_chunk_words.append(word)
        chunk_char_count += word_len + 1  # include space

    # Final chunk (handle remaining words at the end)
    for j in range(chunk_start_idx, len(words_and_times)):
        chunk_words_display = [w for (w, _, _) in words_and_times[chunk_start_idx:j+1]]
        chunk_words_display[-1] = chunk_words_display[-1].upper()  # highlight current word

        subtitle_text = " ".join(chunk_words_display)
        subtitle_start = words_and_times[j][1]
        if j + 1 < len(words_and_times):
            subtitle_end = words_and_times[j+1][1]
        else:
            subtitle_end = words_and_times[j][2]

        subtitles.append(((subtitle_start, subtitle_end), subtitle_text))

    return subtitles

# def make_rolling_subtitled_chunked_video(video_path, words_and_times, output_path, max_words=5):
#     """
#     Loads the video at `video_path`, generates rolling/karaoke-style subtitles
#     from the `words_and_times` data, and writes out to `output_path`.
#     """
#     # 1) Generate the rolling-subtitle intervals for MoviePy
#     sub_data = rolling_subtitle_data_chunked(words_and_times, max_words=max_words)

#     # 2) Define how each subtitle line is rendered
#     def subtitle_generator(txt):
#         """
#         A function that returns a TextClip for the given subtitle text `txt`.
#         This function is called internally by SubtitlesClip for each subtitle segment.
#         """
#         # Create two TextClips - one for the normal text and one for the highlighted word
#         words = txt.split()
#         if not words:
#             return TextClip("", fontsize=50, color='white')
            
#         # Create clips for each word
#         word_clips = []
#         for i, word in enumerate(words):
#             # Last word is highlighted in green and uppercase
#             if i == len(words) - 1:
#                 clip = TextClip(
#                     str(word),  # Convert to string explicitly
#                     fontsize=65,
#                     color='yellow',
#                     font='Comic-Sans-MS-Bold-Italic',
#                     stroke_color='black',
#                     stroke_width=2
#                 )
#             else:
#                 clip = TextClip(
#                     str(word),  # Convert to string explicitly
#                     fontsize=60,
#                     color='#ADD8E6',
#                     font='Comic-Sans-MS-Bold-Italic',
#                     stroke_color='black',
#                     stroke_width=2
#                 )
#             word_clips.append(clip)
        
#         # Calculate total width needed
#         total_width = sum(clip.w for clip in word_clips) + (len(word_clips) - 1) * 10  # 10 pixels spacing
        
#         # Create a blank clip to hold all words
#         final_clip = ColorClip(size=(total_width, word_clips[0].h), color=(0,0,0))
#         final_clip = final_clip.set_opacity(0)  # Make it transparent
        
#         # Position each word
#         x_pos = 0
#         for clip in word_clips:
#             final_clip = CompositeVideoClip([
#                 final_clip,
#                 clip.set_position((x_pos, 0))
#             ])
#             x_pos += clip.w + 10  # Add 10 pixels spacing between words
            
#         return final_clip

#     # 3) Create a SubtitlesClip from our list of timed subtitles
#     subtitles_clip = SubtitlesClip(sub_data, subtitle_generator)

#     # 4) Load the original video
#     video_clip = VideoFileClip(str(video_path))

#     # 5) Composite the video and subtitles together
#     final_clip = CompositeVideoClip([video_clip, subtitles_clip.set_position(('center', 0.85 * video_clip.h))])

#     # 6) Write the final video file
#     final_clip.write_videofile(str(output_path), codec='libx264', fps=video_clip.fps)

#     # Close the clips
#     final_clip.close()
#     video_clip.close()

def make_rolling_subtitled_chunked_video(input_path, srt_data, output_path,
                                 chunk_size=5, y_offset=350, font_size=65,
                                 box_padding=8):
    def hex2rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    video = VideoFileClip(str(input_path))
    W, H = video.size
    base_color      = "#7DF9FF"     # inactive words
    highlight_color = "#00FF41"     # current word
    box_hex       = "#DC143C"
    font = "Comic-Sans-MS-Bold-Italic"
    spacing = 10  # pixels between words
    box_color = hex2rgb(box_hex)

    clips = [video]
    n_chunks = math.ceil(len(srt_data) / chunk_size)

    for ci in range(n_chunks):
        chunk = srt_data[ci*chunk_size:(ci+1)*chunk_size]
        if not chunk:
            break

        # chunk timing
        t0 = chunk[0][1]
        t1 = chunk[-1][2]
        chunk_dur = t1 - t0

        # x start: center the chunk’s total width
        total_w = sum(TextClip(w, fontsize=font_size, font=font).w 
                      for w,_,_ in chunk) \
                  + spacing * (len(chunk)-1)
        x0 = (W - total_w) // 2
        y0 = H - y_offset

        x_cursor = x0

        for word, w_start, w_end in chunk:
            # 1) grey base word, visible for whole chunk
            base_txt = ( TextClip(word, fontsize=font_size, color=base_color, font=font)
                         .set_start(t0)
                         .set_duration(chunk_dur)
                         .set_position((x_cursor, y0)) )
            clips.append(base_txt)

            # 2) red box + green word, at exact same x_cursor
            highlighted  = TextClip(word, fontsize=font_size, color=highlight_color, font=font)
            boxed = ( highlighted 
                      .on_color(
                          size=(highlighted .w + box_padding,
                                highlighted .h + box_padding),
                          color=box_color,   # red
                          col_opacity=1
                      )
                      .set_start(w_start)
                      .set_duration(w_end - w_start)
                      .set_position((x_cursor - box_padding//2,
                                     y0 - box_padding//2)) )
            clips.append(boxed)

            x_cursor += highlighted .w + spacing

    final = CompositeVideoClip(clips).set_duration(video.duration)
    final.write_videofile(str(output_path), fps=video.fps, codec="libx264")
    final.close()

def make_rolling_subtitled_video(video_path, words_and_times, output_path, max_words=5):
    """
    Loads the video at `video_path`, generates rolling/karaoke-style subtitles
    from the `words_and_times` data, and writes out to `output_path`.
    """
    # 1) Generate the rolling-subtitle intervals for MoviePy
    sub_data = rolling_subtitle_data(words_and_times, max_words=max_words)

    # 2) Define how each subtitle line is rendered
    def subtitle_generator(txt):
        """
        A function that returns a TextClip for the given subtitle text `txt`.
        This function is called internally by SubtitlesClip for each subtitle segment.
        """
        # Create two TextClips - one for the normal text and one for the highlighted word
        words = txt.split()
        if not words:
            return TextClip("", fontsize=50, color='white')
            
        # Create clips for each word
        word_clips = []
        for i, word in enumerate(words):
            # Last word is highlighted in green and uppercase
            if i == len(words) - 1:
                clip = TextClip(
                    str(word),  # Convert to string explicitly
                    fontsize=65,
                    color='yellow',
                    font='Comic-Sans-MS-Bold-Italic',
                    stroke_color='black',
                    stroke_width=2
                )
            else:
                clip = TextClip(
                    str(word),  # Convert to string explicitly
                    fontsize=60,
                    color='#ADD8E6',
                    font='Comic-Sans-MS-Bold-Italic',
                    stroke_color='black',
                    stroke_width=2
                )
            word_clips.append(clip)
        
        # Calculate total width needed
        total_width = sum(clip.w for clip in word_clips) + (len(word_clips) - 1) * 10  # 10 pixels spacing
        
        # Create a blank clip to hold all words
        final_clip = ColorClip(size=(total_width, word_clips[0].h), color=(0,0,0))
        final_clip = final_clip.set_opacity(0)  # Make it transparent
        
        # Position each word
        x_pos = 0
        for clip in word_clips:
            final_clip = CompositeVideoClip([
                final_clip,
                clip.set_position((x_pos, 0))
            ])
            x_pos += clip.w + 10  # Add 10 pixels spacing between words
            
        return final_clip

    # 3) Create a SubtitlesClip from our list of timed subtitles
    subtitles_clip = SubtitlesClip(sub_data, subtitle_generator)

    # 4) Load the original video
    video_clip = VideoFileClip(str(video_path))

    # 5) Composite the video and subtitles together
    final_clip = CompositeVideoClip([video_clip, subtitles_clip.set_position(('center', 0.85 * video_clip.h))])

    # 6) Write the final video file
    final_clip.write_videofile(str(output_path), codec='libx264', fps=video_clip.fps)

    # Close the clips
    final_clip.close()
    video_clip.close()

def combine_chars_to_words(csv_path):
    df = pd.read_csv(csv_path)
    chars = df['a'].tolist()
    chars = chars[1:]
    starts = df['b'].tolist()
    starts = starts[1:]
    ends = df['c'].tolist()
    ends = ends[1:]
    words = []
    words_starts = []
    words_ends = []
    word = ""
    start_list = []
    end_list = []
    n = len(chars)
    for i in  range(n):
        char = chars[i]
        start = starts[i]
        end = ends[i]
        if char =="—":
            char = " "
            
        if i == n-1:
            words.append(word)
            start_time = start_list[0]
            end_time = end_list[-1]
            
            words_starts.append(start_time)
            words_ends.append(end_time)
            word = ""
            start_list = []
            end_list = []
        else:
            if char == " ":
                words.append(word)
                start_time = start_list[0]
                end_time = end_list[-1]
                start_list = []
                end_list = []
                words_starts.append(start_time)
                words_ends.append(end_time)
                word = ""
            else:
                word +=char
                start_list.append(start)
                end_list.append(end)
            
        # if i ==n-1:
        #     words.append(word)
        #     start_time = start_list[0]
        #     end_time = end_list[-1]
            
        #     words_starts.append(start_time)
        #     words_ends.append(end_time)
        #     word = ""
        #     start_list = []
        #     end_list = []
    

    df = pd.DataFrame({"words": words, "words_starts": words_starts, "words_ends": words_ends})
    new_csv_path = csv_path.replace(".csv", "_combined.csv")
    df.to_csv(new_csv_path, index=False, encoding="utf-8")
    return words, words_starts, words_ends

def words_to_combined_csv(csv_list,total_duration_video,slide_duration=0.1, fade_duration = 0.1, fps = 24):
    combined_words = []
    combined_words_starts = []
    combined_words_ends = []
    previous_end = 0
    total_duration_prev = 0
    total_duration = 0
    print(f"total csv found: {len(csv_list)}")
    for i in range(len(csv_list)):
        words, words_starts, words_ends = combine_chars_to_words(csv_list[i])
        total_duration_prev += words_ends[-1]
    
    remaining_duration = total_duration_video - total_duration_prev -0.2
    remaining_additional_duration = remaining_duration/(len(csv_list)-1)

    for i in range(len(csv_list)):
        if i == 0:  
            words, words_starts, words_ends = combine_chars_to_words(csv_list[i])
            combined_words.extend(words)
            corrected_words_starts = [start +previous_end +1*(slide_duration +fade_duration) for start in words_starts]
            corrected_words_ends = [end +previous_end +1*(slide_duration +fade_duration) for end in words_ends]
            combined_words_starts.extend(corrected_words_starts)
            combined_words_ends.extend(corrected_words_ends)
            previous_end = corrected_words_ends[-1]
            total_duration_prev += words_ends[-1]
            total_duration += words_ends[-1] + 2*(slide_duration +fade_duration)
        else:
            words, words_starts, words_ends = combine_chars_to_words(csv_list[i])
            combined_words.extend(words)
            corrected_words_starts = [start +previous_end +remaining_additional_duration for start in words_starts]
            corrected_words_ends = [end +previous_end +remaining_additional_duration for end in words_ends]
            combined_words_starts.extend(corrected_words_starts)
            combined_words_ends.extend(corrected_words_ends)
            previous_end = corrected_words_ends[-1]
            total_duration_prev += words_ends[-1]
            total_duration += words_ends[-1] + 2*(slide_duration +fade_duration)

    return combined_words, combined_words_starts, combined_words_ends, total_duration, total_duration_prev


def add_bottom_black_area(clip: VideoFileClip,
                          black_area_height: int = 64):
    """
    Add a black area at the bottom of the video clip (for captions).

    Args:
        clip (VideoFileClip): Video clip to be processed.
        black_area_height (int): Height of the black area.

    Returns:
        VideoFileClip: Processed video clip.
    """
    black_bar = ColorClip(size=(clip.w, black_area_height), color=(255, 255, 255), duration=clip.duration)
    extended_clip = CompositeVideoClip([clip, black_bar.set_position(("center", "bottom"))])
    return extended_clip


def add_zoom_effect(clip, speed=1.0, mode='in', position='center'):
    fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * fps)
    def main(getframe, t):
        frame = getframe(t)
        h, w = frame.shape[: 2]
        i = t * fps
        if mode == 'out':
            i = total_frames - i
        zoom = 1 + (i * ((0.1 * speed) / total_frames))
        positions = {'center':  [(w - (w * zoom)) / 2,  (h - (h  *  zoom)) / 2],
                     'left': [0, (h - (h * zoom)) / 2],
                     'right': [(w - (w * zoom)), (h - (h * zoom)) / 2],
                     'top': [(w - (w * zoom)) / 2, 0],
                     'topleft': [0, 0],
                     'topright': [(w - (w * zoom)), 0],
                     'bottom': [(w - (w * zoom)) / 2, (h - (h * zoom))],
                     'bottomleft': [0, (h - (h * zoom))],
                     'bottomright': [(w - (w * zoom)), (h - (h * zoom))]}
        tx, ty = positions[position]
        M = np.array([[zoom, 0, tx], [0, zoom, ty]])
        frame = cv2.warpAffine(frame, M, (w, h))
        return frame
    return clip.fl(main)


def add_move_effect(clip, direction="left", move_raito=0.9):
    """Add pan left/right effect with dynamic zoom to video clip"""
    w, h = clip.size
    move_distance = int(w * (1 - move_raito))
    
    # Increase zoom factor significantly to ensure no black edges
    
    # required_zoom = 1.3  # Fixed larger zoom
    zoom = (w + move_distance) / w
    def effect(get_frame, t):
        progress = t / clip.duration
        frame = get_frame(t)
        zx = zoom 
        zy = zoom
        tx = (-zoom +1 )*w/2
        ty = (-zoom +1 )*h/2
        z = np.array([[zx, 0, tx], [0, zy, ty]], dtype=np.float32)
        zoomed = cv2.warpAffine(frame, z, (w, h), flags= cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if direction.lower() == "left":
            dx = -move_distance*progress
        else:
            dx = move_distance*(progress)
        M = np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32)
        moved = cv2.warpAffine(zoomed, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return moved
    return clip.fl(effect)


def add_slide_effect(clips, slide_duration=0.4):
    """Concatenate clips with sliding transition"""
    final_clips = []
    for i, clip in enumerate(clips):
        if i > 0:  # Add transition for all clips except first
            clip = clip.crossfadein(slide_duration)
        final_clips.append(clip)
    return concatenate_videoclips(final_clips, method="compose")


def clear_directory(directory_path):
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


def compose_video(story_dir: Union[str, Path],
                  save_path: Union[str, Path],
                  captions: List,
                  music_path: Union[str, Path],
                  num_pages: int,
                  fps: int = 10,
                  audio_sample_rate: int = 16000,
                  audio_codec: str = "mp3",
                  caption_config: dict = {},
                  fade_duration: float = 0.1,
                  slide_duration: float = 0.1,
                  zoom_speed: float = 0.05,
                  move_ratio: float = 0.95,
                  sound_volume: float = 0.0,
                  music_volume: float = 0.3,
                  bg_speech_ratio: float = 0.4):
    try:
        if not isinstance(story_dir, Path):
            story_dir = Path(story_dir)

        sound_dir = story_dir / "sound"
        image_dir = story_dir / "image"
        speech_dir = story_dir / "speech"

        print("\nProcessing video with parameters:")
        print(f"Story directory: {story_dir}")
        print(f"Save path: {save_path}")
        print(f"Number of pages: {num_pages}")
        print(f"FPS: {fps}")
        print(f"Audio sample rate: {audio_sample_rate}")

        csv_list = []
        csv_list_path = "generated_stories\example\speech"
        num_files = 0
        for file in os.listdir(csv_list_path):
            if file.endswith(".csv"):
                num_files +=1
        num_files = num_files
        print(f"Number of csv files: {num_files}")
        for i in range(num_files):
            csv_list.append(csv_list_path + f"\p{i+1}.csv")

        

    
        video_clips = []
        cur_duration = 0
        timestamps = []

        for page in trange(1, num_pages + 1):
            print(f"\nProcessing page {page}")
            
            # Create silence clips
            slide_silence = AudioArrayClip(np.zeros((int(audio_sample_rate * slide_duration), 2)), fps=audio_sample_rate)
            fade_silence = AudioArrayClip(np.zeros((int(audio_sample_rate * fade_duration), 2)), fps=audio_sample_rate)

            # Process speech
            speech_file = speech_dir / f"p{page}.wav"
            if not speech_file.exists():
                print(f"Warning: Speech file not found for page {page}")
                continue

            speech_clip = AudioFileClip(str(speech_file), fps=audio_sample_rate)
            speech_clip = concatenate_audioclips([fade_silence, speech_clip, fade_silence])
            
            # Add slide silence
            if page == 1:
                speech_clip = concatenate_audioclips([speech_clip, slide_silence])
            else:
                speech_clip = concatenate_audioclips([slide_silence, speech_clip, slide_silence])

            # Add timestamp
            timestamps.append([cur_duration + fade_duration,
                             cur_duration + speech_clip.duration - fade_duration - slide_duration])
            cur_duration += speech_clip.duration - slide_duration

            # Process image
            image_file = image_dir / f"p{page}.png"
            if not image_file.exists():
                print(f"Warning: Image file not found for page {page}")
                continue

            image_clip = ImageClip(str(image_file))
            image_clip = image_clip.set_duration(speech_clip.duration).set_fps(fps)
            image_clip = image_clip.crossfadein(fade_duration).crossfadeout(fade_duration)

            # Add effects
            if random.random() <= 0.5:
                zoom_mode = "in" if random.random() <= 0.5 else "out"
                image_clip = add_zoom_effect(image_clip, zoom_speed, zoom_mode)
            else:
                direction = "left" if random.random() <= 0.5 else "right"
                image_clip = add_move_effect(image_clip, direction=direction, move_raito=move_ratio)

            # Process sound
            sound_file = sound_dir / f"p{page}.wav"
            audio_clip = speech_clip  # Default to speech only
            if sound_file.exists():
                try:
                    sound_clip = AudioFileClip(str(sound_file), fps=audio_sample_rate)
                    sound_clip = sound_clip.audio_fadein(fade_duration)
                    if sound_clip.duration < speech_clip.duration:
                        sound_clip = audio_loop(sound_clip, duration=speech_clip.duration)
                    else:
                        sound_clip = sound_clip.subclip(0, speech_clip.duration)
                    audio_clip = CompositeAudioClip([speech_clip, sound_clip.volumex(sound_volume)])
                except Exception as e:
                    print(f"Error processing sound for page {page}: {str(e)}")

            video_clip = image_clip.set_audio(audio_clip)
            video_clips.append(video_clip)
            print(f"Successfully created video clip for page {page}")

        if not video_clips:
            print("Error: No video clips were created")
            return None

        print(f"\nSuccessfully created {len(video_clips)} video clips")
        print("\nComposing final video...")

        composite_clip = add_slide_effect(video_clips, slide_duration=slide_duration)
        # composite_clip = add_bottom_black_area(composite_clip, black_area_height=caption_config.get("area_height", 100))
        composite_clip1 = composite_clip
        composite_clip2 = composite_clip
        # Create a copy of caption_config to avoid modifying the original
        caption_config_copy = caption_config.copy()
        max_caption_length = caption_config_copy.pop("max_length", 100)
        area_height = caption_config_copy.pop("area_height", 100)
        
        # Ensure required caption parameters are set
        caption_config_copy.update({
            "font": caption_config_copy.get("font", "Arial"),
            "fontsize": caption_config_copy.get("fontsize", 12),
            "color": caption_config_copy.get("color", "white"),
            "stroke_color": caption_config_copy.get("stroke_color", "black"),
            "stroke_width": caption_config_copy.get("stroke_width", 2),
            "area_height": area_height
        })
        
        composite_clip = add_caption(
            captions,
            story_dir / "captions.srt",
            timestamps,
            composite_clip,
            max_caption_length,
            **caption_config_copy
        )

        
        # Add music if available

        if music_path and Path(music_path).exists():
            music_clip = AudioFileClip(str(music_path), fps=audio_sample_rate)
            if music_clip.duration < composite_clip.duration:
                music_clip = audio_loop(music_clip, duration=composite_clip.duration)
            else:
                music_clip = music_clip.subclip(0, composite_clip.duration)
            all_audio_clip = CompositeAudioClip([composite_clip.audio, music_clip.volumex(music_volume)])
            composite_clip = composite_clip.set_audio(all_audio_clip)
            composite_clip1 = composite_clip1.set_audio(all_audio_clip)
        
        save_path1 = Path(str(save_path).replace(".mp4", "_without_subtitles.mp4"))
        print("About to write:", save_path1.resolve())
        save_path2 = Path(str(save_path).replace(".mp4", "_without_music_subtitles.mp4"))
        print("About to write:", save_path2.resolve())
        print(f"\nWriting video to {save_path2}")
        print(f"\nWriting video to {save_path1}")
        print(f"\nWriting video to {save_path}")
        try:
            composite_clip1.write_videofile(
                str(save_path1),
                fps=fps,
                codec='libx264',
                audio_codec=audio_codec,
                audio_fps=audio_sample_rate,
                preset='ultrafast',
                threads=4,
                ffmpeg_params=["-pix_fmt", "yuv420p"]
            )
            composite_clip2.write_videofile(
                str(save_path2),
                fps=fps,
                codec='libx264',
                audio_codec=audio_codec,
                audio_fps=audio_sample_rate,
                preset='ultrafast',
                threads=4,
                ffmpeg_params=["-pix_fmt", "yuv420p"]
            )
            composite_clip.write_videofile(
                str(save_path),
                fps=fps,
                codec='libx264',
                audio_codec=audio_codec,
                audio_fps=audio_sample_rate,
                preset='ultrafast',
                threads=4,
                ffmpeg_params=["-pix_fmt", "yuv420p"]
            )
            input_video = save_path1
            video = VideoFileClip(str(input_video))
            duration = video.duration 
            words, words_starts, words_ends, total_duration, total_duration_prev = words_to_combined_csv(csv_list, duration)
            data = pd.DataFrame({"words": words, "words_starts": words_starts, "words_ends": words_ends})
            srt_data = list(zip(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]))
            output_video_path = Path(str(save_path).replace(".mp4", "rolling_subs.mp4"))  
            ensure_imagemagick()
            make_rolling_subtitled_video(input_video, srt_data, output_video_path, max_words=5)
            print("Done! for rolling subtitles Check:", output_video_path)
            output_video_path_chunked = Path(str(output_video_path).replace("rolling_subs.mp4", "chunked_rolling_subs.mp4"))
            
            make_rolling_subtitled_chunked_video(input_video, srt_data, output_video_path_chunked, chunk_size=4)
            print("Done for chunked rolling subtitles video: ", output_video_path_chunked)

            if Path(output_video_path_chunked).exists() and Path(output_video_path).exists():
                print(f"Video file successfully written to {output_video_path_chunked}")
                print(f"Video file successfully written to {output_video_path}")
                return composite_clip
            else:
                print(f"Error: Video file was not created at {save_path}")
                return None
            
                
        except Exception as e:
            print(f"Error writing video file: {str(e)}")
            traceback.print_exc()
            return None

        
     
        
        

    except Exception as e:
        print(f"Error in video composition: {str(e)}")
        traceback.print_exc()
        return None


@register_tool("slideshow_video_compose")
class SlideshowVideoComposeAgent:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    def call(self, params):
        try:
            story_dir = Path(params["story_dir"])
            save_path = params["save_path"]
            captions = params["captions"]
            music_path = params["music_path"]
            num_pages = params["num_pages"]
            
            print(f"\nStarting video composition with:")
            print(f"Story directory: {story_dir}")
            print(f"Save path: {save_path}")
            print(f"Number of pages: {num_pages}")
            print(f"Captions: {captions}")
            
            # Get configuration parameters with defaults
            config = {
                "fps": self.cfg.get("fps", 10),
                "audio_sample_rate": self.cfg.get("audio_sample_rate", 16000),
                "audio_codec": self.cfg.get("audio_codec", "mp3"),
                "fade_duration": self.cfg.get("fade_duration", 0.1),
                "slide_duration": self.cfg.get("slide_duration", 0.1),
                "zoom_speed": self.cfg.get("zoom_speed", 0.5),
                "move_ratio": self.cfg.get("move_ratio", 0.95),
                "sound_volume": self.cfg.get("sound_volume", 0.2),
                "music_volume": self.cfg.get("music_volume", 0.2),
                "bg_speech_ratio": self.cfg.get("bg_speech_ratio", 0.4),
                "caption_config": {
                    "font": self.cfg.get("caption", {}).get("font", "Arial"),
                    "fontsize": self.cfg.get("caption", {}).get("fontsize", 24),  # Increased font size
                    "color": "white",  # Force white color
                    "stroke_color": "black",  # Force black stroke
                    "stroke_width": 2,  # Increased stroke width for better visibility
                    "area_height": self.cfg.get("caption", {}).get("area_height", 120),
                    "max_length": self.cfg.get("caption", {}).get("max_length", 100)
                }
            }
            
            print(f"Caption configuration: {config['caption_config']}")
            
            # Create temporary directory for intermediate files
            temp_dir = Path(save_path).parent / "temp"
            temp_dir.mkdir(exist_ok=True, parents=True)
            
            try:
                result = compose_video(
                    story_dir=story_dir,
                    save_path=save_path,
                    captions=captions,
                    music_path=music_path,
                    num_pages=num_pages,
                    **config
                )
                
                if result is not None:
                    print(f"Video successfully saved to: {save_path}")
                    return result
                else:
                    print("Error: Video composition failed")
                    return None
                    
            finally:
                # Clean up temporary directory
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up temp directory: {e}")
                
        except Exception as e:
            print(f"Error in video composition: {str(e)}")
            import traceback
            traceback.print_exc()
            return None