from typing import Dict, Callable
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from openai import OpenAI

from mm_story_agent.base import register_tool
# from mm_story_agent.utils.llm_logger import llm_logger

# Load environment variables from .env file
load_dotenv()

print("Registering LLM agents...")

# @register_tool("gemini")
# class GeminiAgent(object):

#     def __init__(self,
#                  config: Dict):
#         print("Initializing GeminiAgent")
#         self.system_prompt = config.get("system_prompt")
#         track_history = config.get("track_history", False)
#         if self.system_prompt is None:
#             self.history = []
#         else:
#             self.history = [
#                 {"role": "system", "content": self.system_prompt}
#             ]
#         self.track_history = track_history
        
#         # Initialize Gemini with API key from .env
#         api_key = os.getenv('GOOGLE_API_KEY')
#         if not api_key:
#             raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-2.0-flash')
    
#     def basic_success_check(self, response):
#         if not response or not response.text:
#             print(response)
#             return False
#         else:
#             return True
    
#     def call(self,
#              prompt: str,
#              model_name: str = "gemini-2.0-flash",
#              top_p: float = 0.95,
#              temperature: float = 1.0,
#              seed: int = 1,
#              max_length: int = 1024,
#              max_try: int = 5,
#              success_check_fn: Callable = None
#              ):
#         self.history.append({
#             "role": "user",
#             "content": prompt
#         })
#         success = False
#         try_times = 0
#         response_text = ""
        
#         while try_times < max_try:
#             try:
#                 # Format messages for Gemini
#                 messages = []
#                 for msg in self.history:
#                     if msg["role"] == "system":
#                         messages.append(f"System: {msg['content']}")
#                     elif msg["role"] == "user":
#                         messages.append(f"User: {msg['content']}")
#                     elif msg["role"] == "assistant":
#                         messages.append(f"Assistant: {msg['content']}")
                
#                 # Add JSON formatting instruction for certain prompts
#                 if success_check_fn and "json_parse_outline" in str(success_check_fn):
#                     messages.append("""
# Please ensure the response is in valid JSON format with exactly this structure:
# {
#     "story_title": "title here",
#     "story_outline": [
#         {"chapter_title": "chapter 1 title", "chapter_summary": "chapter 1 summary"},
#         {"chapter_title": "chapter 2 title", "chapter_summary": "chapter 2 summary"}
#     ]
# }
# Wrap the response in ```json``` tags.
# """)
                
#                 # Add special handling for list responses
#                 if success_check_fn and "parse_list" in str(success_check_fn):
#                     messages.append("""
# Please ensure the response is a Python list of strings, each string being a page of the story.
# Format the response like this:
# [
#     "First page content here.",
#     "Second page content here.",
#     "Third page content here."
# ]
# Wrap the response in ```python``` tags.
# """)
                
#                 # Generate response
#                 response = self.model.generate_content(
#                     "\n".join(messages),
#                     generation_config=genai.types.GenerationConfig(
#                         temperature=temperature,
#                         top_p=top_p,
#                         max_output_tokens=max_length,
#                     )
#                 )
                
#                 if success_check_fn is None:
#                     success_check_fn = lambda x: True
                
#                 response_text = response.text.strip()
                
#                 # Handle JSON responses
#                 if success_check_fn and "json_parse_outline" in str(success_check_fn):
#                     # Extract JSON content if wrapped in ```json``` tags
#                     if "```json" in response_text:
#                         response_text = response_text.split("```json")[-1].split("```")[0].strip()
#                     # Try to validate JSON
#                     try:
#                         json.loads(response_text)
#                     except json.JSONDecodeError:
#                         try_times += 1
#                         continue
                
#                 # Handle list responses
#                 if success_check_fn and "parse_list" in str(success_check_fn):
#                     if "```python" in response_text:
#                         response_text = response_text.split("```python")[-1].split("```")[0].strip()
#                     try:
#                         # Validate that it's a valid Python list
#                         eval_result = eval(response_text)
#                         if not isinstance(eval_result, list) or not all(isinstance(x, str) for x in eval_result):
#                             try_times += 1
#                             continue
#                     except Exception:
#                         try_times += 1
#                         continue
                
#                 if self.basic_success_check(response) and success_check_fn(response_text):
#                     self.history.append({
#                         "role": "assistant",
#                         "content": response_text
#                     })
#                     success = True
#                     break
#                 else:
#                     try_times += 1
                    
#             except Exception as e:
#                 print(f"Error in Gemini API call: {e}")
#                 try_times += 1
        
#         if not self.track_history:
#             if self.system_prompt is not None:
#                 self.history = self.history[:1]
#             else:
#                 self.history = []
        
#         # Log the LLM call
#         metadata = {
#             "model_name": model_name,
#             "temperature": temperature,
#             "top_p": top_p,
#             "max_length": max_length,
#             "try_times": try_times,
#             "success_check_fn": str(success_check_fn) if success_check_fn else None
#         }
#         # llm_logger.log_call(
#         #     llm_type="gemini",
#         #     prompt=prompt,
#         #     response=response_text if success else "",
#         #     success=success,
#         #     metadata=metadata
#         # )
        
#         return response_text if success else "", success
   
@register_tool("openai")
class OpenAIAgent(object):
    def __init__(self, config: Dict):
        print("Initializing OpenAIAgent")
        self.system_prompt = config.get("system_prompt")
        track_history = config.get("track_history", False)
        if self.system_prompt is None:
            self.history = []
        else:
            self.history = [
                {"role": "system", "content": self.system_prompt}
            ]
        self.track_history = track_history
        
        # Initialize OpenAI with API key from .env
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model = config.get("model_name", "gpt-4.1-2025-04-14")
    
    def basic_success_check(self, response):
        if not response:
            return False
        return True
    
    def call(self,
             prompt: str,
             model_name: str = None,
             temperature: float = 1.0,
             seed: int = 1,
             max_length: int = 1024,
             max_try: int = 5,
             success_check_fn: Callable = None):
        
        self.history.append({
            "role": "user",
            "content": prompt
        })
        
        print(f"Prompt: {prompt}")
        success = False
        try_times = 0
        result = ""
        error_message = ""
        
        while try_times < max_try:
            try:
                messages = []
                for msg in self.history:
                    if msg["role"] == "system":
                        messages.append({"role": "system", "content": msg["content"]})
                    elif msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        messages.append({"role": "assistant", "content": msg["content"]})
                
                # Add JSON formatting instruction for certain prompts
                if success_check_fn and "json_parse_outline" in str(success_check_fn):
                    messages.append({
                        "role": "system",
                        "content": """
Please ensure the response is in valid JSON format with exactly this structure:
{
    "story_title": "title here",
    "story_outline": [
        {"chapter_title": "chapter 1 title", "chapter_summary": "chapter 1 summary"},
        {"chapter_title": "chapter 2 title", "chapter_summary": "chapter 2 summary"}
    ]
}
Wrap the response in ```json``` tags.
"""
                    })
                
                # Add special handling for list responses
                if success_check_fn and "parse_list" in str(success_check_fn):
                    messages.append({
                        "role": "system",
                        "content": """
Please ensure the response is a Python list of strings, each string being a page of the story.
Format the response like this:
[
    "First page content here.",
    "Second page content here.",
    "Third page content here."
]
Wrap the response in ```python``` tags.
"""
                    })
                
                response = self.client.chat.completions.create(
                    model=model_name or self.model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_length
                )
                
                result = response.choices[0].message.content.strip()
                
                if success_check_fn is None:
                    success_check_fn = lambda x: True
                
                # Handle JSON responses
                if success_check_fn and "json_parse_outline" in str(success_check_fn):
                    # Extract JSON content if wrapped in ```json``` tags
                    if "```json" in result:
                        result = result.split("```json")[-1].split("```")[0].strip()
                    # Try to validate JSON
                    try:
                        json.loads(result)
                    except json.JSONDecodeError:
                        try_times += 1
                        continue
                
                # Handle list responses
                if success_check_fn and "parse_list" in str(success_check_fn):
                    if "```python" in result:
                        result = result.split("```python")[-1].split("```")[0].strip()
                    try:
                        # Validate that it's a valid Python list
                        eval_result = eval(result)
                        if not isinstance(eval_result, list) or not all(isinstance(x, str) for x in eval_result):
                            try_times += 1
                            continue
                    except Exception:
                        try_times += 1
                        continue
                
                if self.basic_success_check(result) and success_check_fn(result):
                    self.history.append({
                        "role": "assistant",
                        "content": result
                    })
                    success = True
                    break
                else:
                    try_times += 1
                    
            except Exception as e:
                error_message = str(e)
                print(f"Error in OpenAI call: {e}")
                try_times += 1
        
        if not self.track_history:
            if self.system_prompt is not None:
                self.history = self.history[:1]
            else:
                self.history = []
        
        # Log the LLM call with enhanced details
        metadata = {
            "model_name": model_name or self.model,
            "temperature": temperature,
            "seed": seed,
            "max_length": max_length,
            "try_times": try_times,
            "success_check_fn": str(success_check_fn) if success_check_fn else None,
            "system_prompt": self.system_prompt
        }
        
        # llm_logger.log_call(
        #     llm_type="openai",
        #     prompt=prompt,
        #     response=result if success else "",
        #     success=success,
        #     prompt_type="story_generation",
        #     metadata=metadata,
        #     error_message=error_message if not success else None,
        #     retry_count=try_times,
        #     model_name=model_name or self.model,
        #     temperature=temperature,
        #     max_tokens=max_length
        # )
        
        return result if success else "", success
   