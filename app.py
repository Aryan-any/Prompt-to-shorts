import gradio as gr
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import re
import json
import os
from pathlib import Path
from datetime import datetime
import time
import asyncio
from typing import Dict, List

# Your existing imports
from mm_story_agent import MMStoryAgent
from mm_story_agent.utils.llm_output_check import parse_list
from mm_story_agent.base import register_tool, init_tool_instance
from mm_story_agent.prompts_en import base_story_to_video_specs
from mm_story_agent.utils.llm_utils import get_llm_config

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# =================== CONFIGURATION ===================
SHEET_URL = 'https://docs.google.com/spreadsheets/d/10X5H1vdzVz5UxSdyvhk-RMyJ3NUlS-ucUN-lKllRkg8/edit?resourcekey=&gid=425272253#gid=425272253'
SERVICE_ACCOUNT_FILE = 'allinai_service_account.json'
YOUTUBE_TOKEN_FILE = 'token_youtube1.json'
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# =================== GOOGLE SHEETS SETUP ===================
def init_google_sheets():
    """Initialize Google Sheets connection"""
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SHEET_URL).sheet1
    return sheet, creds

def get_pending_videos(sheet):
    """Get videos that need to be made"""
    headers = ["Timestamp", "Base Story", "Genre",
              "Ready to make", "language",
              "Youtube Link", "done?"]
    
    data = sheet.get_all_records(expected_headers=headers)
    raw_headers = sheet.row_values(1)
    print("Raw headers with repr:", [repr(h) for h in raw_headers])
    # data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    
    # Find videos where "Ready to make" is "Yes" and "done?" is not "yes"
    pending = df[
        (df['Ready to make'].str.lower() == 'yes') & 
        (df['done?'].str.lower() != 'yes')
    ]
    
    return pending, df

# def get_pending_videos(sheet):
#     try:
        
#         data = sheet.get_all_records()
#         df = pd.DataFrame(data)
#     except Exception as e:
#         raw_headers = sheet.row_values(1)
#         from collections import Counter 
#         counts = Counter(raw_headers)
#         dupes = [h for h, c in counts.items() if c > 1]
#         print(f" header rows with duplicates: {dupes}\n")
#         print(f" Available headers: {raw_headers}\n")
#     pending = df[
#         (df['Ready to make'].str.lower() == 'yes') & 
#         (df['done?'].str.lower() != 'yes')
#     ]
#     return pending, df
# # =================== YOUR EXISTING FUNCTIONS ===================

config_params_system = """
You are an expert in helping extracting the story_topic, main_role, scene from the given base story.
the ouput should be strictly a json in the following format:
{
"story_topic" = "xxxx",
"main_role" = "xxxx",
"scene" = "xxxx"
}
the extracted fields must be in accordance to the given input base story.
""".strip()

json_reformatter_system = """
You are an expert in formatting the given input string in a perfect json ready to be parsed.
given the input:
{init_input}
you have to format it in json ready to be parsed, you just have to correct the structure do not tamper the data inside it.
"""

def json_reformatter(data_in):
    json_config = {
       "tool": "openai",
        "cfg": {
            "system_prompt": json_reformatter_system.format(init_input=data_in),
            "track_history": False
        }
    }
    json_tool = init_tool_instance(get_llm_config(json_config))
    correct_json = json_tool.call(json_reformatter_system.format(init_input=data_in))
    return correct_json

def get_config_params(base_story):
    params_config = {
       "tool": "openai",
        "cfg": {
            "system_prompt": config_params_system,
            "track_history": False
        }
    }
    params_tool = init_tool_instance(get_llm_config(params_config))
    config_params = params_tool.call(base_story)
    return config_params

def get_video_metadata(base_story, lang):
    metadata_config = {
        "tool": "openai",
        "cfg": {
            "system_prompt": base_story_to_video_specs.format(base_story=base_story, lang=lang),
            "track_history": False
        }
    }
    metadata_tool = init_tool_instance(get_llm_config(metadata_config))
    video_metadata = metadata_tool.call(base_story_to_video_specs.format(base_story=base_story, lang=lang))
    return video_metadata

def upload_video_to_youtube(video_file_path, title, description, tags=None, privacy_status="public"):
    """Upload video to YouTube"""
    if os.path.exists(YOUTUBE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(YOUTUBE_TOKEN_FILE, SCOPES)
    else:
        raise Exception("YouTube token file not found")
    
    youtube = build('youtube', 'v3', credentials=creds)
    
    request_body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags if tags else []
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }
    
    media = MediaFileUpload(video_file_path, chunksize=-1, resumable=True, mimetype='video/*')
    request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=media
    )
    
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")
    
    return response['id']

# =================== VIDEO PROCESSING PIPELINE ===================

# Your existing config template
config_template = {
    "sample_rate": 16000,
    "image_height": 512,
    "image_width": 512,
    "story_dir": "generated_stories/example",
    
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
            "video_style": "youtube shorts",
        },
    },
    
    "speech_generation": {
        "tool": "elevenlabs",
        "cfg": {
            "sample_rate": 16000,
        },
        "params": {},
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
        "params": {
            "style_name": "Storybook",
            "quality": "standard",
        },
    },
    
    "music_generation": {
        "tool": "musicgen_t2m",
        "cfg": {
            "device": "cpu",
            "llm_type": "gemini",
            "num_turns": 3,
            "model_name": "facebook/musicgen-small",
        },
        "params": {
            "duration": 30,
        },
    },
    
    "video_compose": {
        "tool": "slideshow_video_compose",
        "cfg": {},
        "params": {
            "height": 512,
            "width": 512,
            "story_dir": "generated_stories/example",
            "fps": 8,
            "audio_sample_rate": 16000,
            "audio_codec": "mp3",
            "caption": {
                "font": "resources/font/msyh.ttf",
                "fontsize": 32,
                "color": "white",
                "max_length": 50,
            },
            "slideshow_effect": {
                "bg_speech_ratio": 0.6,
                "sound_volume": 0.5,
                "music_volume": 0.4,
                "fade_duration": 0.1,
                "slide_duration": 0.1,
                "zoom_speed": 0.5,
                "move_ratio": 0.9,
            },
        },
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

def process_single_video(row_data, sheet, row_index, progress_callback=None):
    """Process a single video following your exact workflow"""
    
    base_story = row_data['Base Story']
    language = row_data['language']
    
    # Generate unique video name
    now = datetime.now()
    video_name = now.strftime("%Y%m%d_%H%M%S")
    
    max_tries = 5
    
    # Step 1: Get video metadata
    if progress_callback:
        progress_callback(f"Getting metadata for: {base_story[:50]}...")
    
    for attempt in range(1, max_tries + 1):
        try:
            metadata = get_video_metadata(base_story, lang=language)
            init_json_string_with_markers = metadata[0]
            json_string_with_markers = json_reformatter(init_json_string_with_markers)[0]
            json_string = re.sub(r'```', '', json_string_with_markers)
            json_string = re.sub(r'```', '', json_string).strip()           
            parsed_metadata = json.loads(json_string)
            break
        except json.JSONDecodeError as e:
            if attempt == max_tries:
                raise Exception(f"Failed to parse metadata after {max_tries} attempts")
            else:
                continue
    
    video_title = parsed_metadata['video_title']
    video_description = parsed_metadata['video_description']
    video_tags = parsed_metadata['video_tags']
    video_description += " #Shorts"
    
    for tag in video_tags:
        video_description += " #" + tag
    
    # Step 2: Get config parameters
    if progress_callback:
        progress_callback(f"Extracting story parameters...")
    
    for attempt in range(1, max_tries + 1):
        try:
            params = get_config_params(base_story)
            init_json_string_with_markers = params[0]
            json_string_with_markers = json_reformatter(init_json_string_with_markers)[0]
            json_string = re.sub(r'```', '', json_string_with_markers)
            json_string = re.sub(r'```', '', json_string).strip()
            parsed_params = json.loads(json_string)
            break
        except json.JSONDecodeError as e:
            if attempt == max_tries:
                raise Exception(f"Failed to parse parameters after {max_tries} attempts")
            else:
                continue
    
    story_topic = parsed_params['story_topic']
    main_role = parsed_params['main_role']
    scene = parsed_params['scene']
    
    # Step 3: Configure and generate video
    if progress_callback:
        progress_callback(f"Generating video: {video_title}")
    
    config = config_template.copy()
    config["story_writer"]["params"]["story_topic"] = story_topic
    config["story_writer"]["params"]["main_role"] = main_role
    config["story_writer"]["params"]["scene"] = scene
    config["story_writer"]["params"]["base_story"] = base_story
    config["story_writer"]["params"]["story_lang"] = language
    config["speech_generation"]["params"]["lang"] = language
    
    # Generate video
    agent = MMStoryAgent(low_memory_mode=True)
    agent.call(config, video_name)
    
    # Step 4: Upload to YouTube
    if progress_callback:
        progress_callback(f"Uploading to YouTube...")
    
    base_dir = Path("generated_stories") / "example"
    if language == 'english':
        file_name = f"{video_name}chunked_rolling_subs.mp4"
    elif language == 'hindi':
        file_name = f"{video_name}_without_subtitles.mp4"
    else:
        file_name = f"{video_name}chunked_rolling_subs.mp4"  # default
    
    video_path = base_dir / file_name
    
    if not video_path.exists():
        raise Exception(f"Video file not found: {video_path}")
    
    video_id = upload_video_to_youtube(
        video_file_path=str(video_path),
        title=video_title,
        description=video_description,
        privacy_status="public"
    )
    
    # Step 5: Update Google Sheet
    if progress_callback:
        progress_callback(f"Updating Google Sheet...")
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Update YouTube Link column (F)
    sheet.update_cell(row_index + 2, 6, video_url)  # +2 because of header and 0-indexing
    
    # Update done? column (G)
    sheet.update_cell(row_index + 2, 7, "yes")
    
    if progress_callback:
        progress_callback(f"✅ Completed: {video_title}")
    
    return {
        "video_title": video_title,
        "video_url": video_url,
        "video_id": video_id
    }

# =================== GRADIO INTERFACE ===================

async def batch_process_videos(progress=gr.Progress()):
    """Process all pending videos"""
    
    try:
        sheet, creds = init_google_sheets()
        pending_df, full_df = get_pending_videos(sheet)
        
        total_videos = len(pending_df)
        
        if total_videos == 0:
            return "🎉 No pending videos to process!"
        
        progress(0, desc=f"Starting batch processing of {total_videos} videos...")
        
        results = []
        
        for i, (index, row) in enumerate(pending_df.iterrows()):
            current_progress = (i + 1) / total_videos
            
            def update_progress(msg):
                progress(current_progress, desc=f"({i+1}/{total_videos}) {msg}")
            
            try:
                result = process_single_video(row, sheet, index, update_progress)
                results.append(f"✅ {result['video_title']} - {result['video_url']}")
                
            except Exception as e:
                error_msg = f"❌ Failed to process story {i+1}: {str(e)}"
                results.append(error_msg)
                print(error_msg)
        
        progress(1.0, desc="Batch processing completed!")
        
        summary = f"## Batch Processing Complete\n\n"
        summary += f"**Total processed:** {total_videos}\n\n"
        summary += "### Results:\n"
        for result in results:
            summary += f"- {result}\n"
        
        return summary
        
    except Exception as e:
        return f"❌ Error during batch processing: {str(e)}"

def refresh_pending_count():
    """Get current count of pending videos"""
    try:
        sheet, _ = init_google_sheets()
        pending_df, _ = get_pending_videos(sheet)
        return len(pending_df)
    except Exception as e:
        return f"Error: {str(e)}"

def get_pending_stories_preview():
    """Get preview of pending stories"""
    try:
        sheet, _ = init_google_sheets()
        pending_df, _ = get_pending_videos(sheet)
        
        if len(pending_df) == 0:
            return "No pending stories found."
        
        preview = "## Pending Stories Preview\n\n"
        
        for i, (index, row) in enumerate(pending_df.head(10).iterrows()):
            story_preview = row['Base Story'][:100] + "..." if len(row['Base Story']) > 100 else row['Base Story']
            preview += f"**{i+1}.** {row['Genre']} ({row['language']})\n"
            preview += f"*{story_preview}*\n\n"
        
        if len(pending_df) > 10:
            preview += f"... and {len(pending_df) - 10} more stories\n"
        
        return preview
        
    except Exception as e:
        return f"Error loading preview: {str(e)}"

# =================== GRADIO APP ===================

def create_gradio_app():
    """Create the Gradio application"""
    
    with gr.Blocks(title="🎬 AI Video Agent Control Panel", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🎬 AI Video Agent Control Panel")
        gr.Markdown("Control your MMStoryAgent video generation pipeline")
        
        with gr.Row():
            with gr.Column(scale=1):
                pending_count = gr.Number(
                    label="📊 Videos Pending", 
                    value=0, 
                    interactive=False
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Count", variant="secondary")
                    preview_btn = gr.Button("👁️ Preview Stories", variant="secondary")
                
                start_btn = gr.Button(
                    "🚀 Start Video Production", 
                    variant="primary", 
                    size="lg"
                )
                
            with gr.Column(scale=2):
                status_box = gr.Markdown("Ready to process videos...")
                
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                preview_box = gr.Markdown("Click 'Preview Stories' to see pending stories")
        
        # Event handlers
        refresh_btn.click(
            fn=refresh_pending_count,
            outputs=pending_count
        )
        
        preview_btn.click(
            fn=get_pending_stories_preview,
            outputs=preview_box
        )
        
        start_btn.click(
            fn=batch_process_videos,
            outputs=status_box
        ).then(
            fn=refresh_pending_count,
            outputs=pending_count
        )
        
        # Auto-refresh on load
        demo.load(
            fn=refresh_pending_count,
            outputs=pending_count
        )
        
        demo.load(
            fn=get_pending_stories_preview,
            outputs=preview_box
        )
    
    return demo

# =================== MAIN ===================

if __name__ == "__main__":
    app = create_gradio_app()
    app.queue().launch(
        server_name="localhost",
        server_port=7000,
        share=True
    )
