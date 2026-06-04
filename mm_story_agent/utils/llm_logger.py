import json
import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

class LLMLogger:
    def __init__(self, log_dir: str = "logs/llm_calls"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_call(self, 
                 llm_type: str,
                 prompt: str,
                 response: str,
                 success: bool,
                 prompt_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 asker_prompt: Optional[str] = None,
                 expert_prompt: Optional[str] = None,
                 image_prompts: Optional[List[str]] = None,
                 story_pages: Optional[List[str]] = None,
                 error_message: Optional[str] = None,
                 retry_count: Optional[int] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        """Log an LLM call with its details.
        
        Args:
            llm_type: Type of LLM used (e.g., "gemini", "openai")
            prompt: The input prompt sent to the LLM
            response: The response received from the LLM
            success: Whether the call was successful
            prompt_type: Type of system being used (e.g., "image_review", "question_asker", "expert_system")
            metadata: Additional metadata about the call
            asker_prompt: The prompt used by the asker agent
            expert_prompt: The prompt used by the expert agent
            image_prompts: List of prompts used for image generation
            story_pages: List of story pages if applicable
            error_message: Error message if the call failed
            retry_count: Number of retries attempted
            model_name: Name of the model used
            temperature: Temperature setting used
            max_tokens: Maximum tokens used
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "llm_type": llm_type,
            "prompt_type": prompt_type,
            "prompt": prompt,
            "response": response,
            "success": success,
            "metadata": json.dumps(metadata or {}),
            "asker_prompt": asker_prompt or "",
            "expert_prompt": expert_prompt or "",
            "image_prompts": json.dumps(image_prompts or []),
            "story_pages": json.dumps(story_pages or []),
            "error_message": error_message or "",
            "retry_count": retry_count or 0,
            "model_name": model_name or "",
            "temperature": temperature or 0.0,
            "max_tokens": max_tokens or 0
        }
        
        # Create a log file for the current date
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_file = self.log_dir / f"llm_calls_{date_str}.csv"
        
        # Check if file exists to determine if we need to write headers
        file_exists = log_file.exists()
        
        # Append the log entry to the file
        with open(log_file, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

# Create a global logger instance
llm_logger = LLMLogger() 